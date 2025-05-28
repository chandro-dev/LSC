# MODELO MEJORADO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logp = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class SignRecognitionGRU(nn.Module):
    def __init__(self, input_dim, num_classes=50, hidden_dim=64, num_layers=1):
        super(SignRecognitionGRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        output, _ = self.rnn(x)  # (batch, seq_len, hidden_dim*2)
        pooled = torch.mean(output, dim=1)  # Global average pooling
        return self.classifier(pooled)



from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model_improved(model, train_dataset, val_loader: DataLoader = None, 
                        num_epochs=50, lr=1e-3, save_path="sign_language_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # === Sampler por clase para mejorar el balance ===
    all_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_sample_count = np.array([all_labels.count(t) for t in all_labels])
    class_weights = 1. / class_sample_count
    samples_weights = np.array([class_weights[t] for t in all_labels])
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # === DataLoader con sampler ===
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)

    # === Focal Loss con pesos por clase ===
    unique_labels = np.unique(all_labels)
    weights_ce = compute_class_weight('balanced', classes=unique_labels, y=all_labels)
    weights_ce = torch.FloatTensor(weights_ce).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weights_ce)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            total_samples += batch_x.size(0)

        scheduler.step()
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}% - LR: {scheduler.get_last_lr()[0]:.6f}")

        # === Validación (igual que antes)
        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            class_correct = {}
            class_total = {}

            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    val_outputs = model(val_x)
                    val_pred = val_outputs.argmax(dim=1)
                    val_correct += (val_pred == val_y).sum().item()
                    val_total += val_y.size(0)

                    for label in val_y.cpu().numpy():
                        class_total[label] = class_total.get(label, 0) + 1

                    for i, label in enumerate(val_y.cpu().numpy()):
                        if val_pred[i].cpu().item() == label:
                            class_correct[label] = class_correct.get(label, 0) + 1

            val_acc = val_correct / val_total * 100
            print(f"           Val Accuracy: {val_acc:.2f}%")

            if epoch % 10 == 0:
                print("Accuracy por clase:")
                for class_id in sorted(class_total.keys()):
                    acc = (class_correct.get(class_id, 0) / class_total[class_id]) * 100
                    print(f"  Clase {class_id}: {acc:.1f}% ({class_correct.get(class_id, 0)}/{class_total[class_id]})")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping en epoch {epoch+1}")
                break

    print(f"\n✅ Mejor modelo guardado en: {save_path} (Val Acc: {best_val_acc:.2f}%)")
    return model
