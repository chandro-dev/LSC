from transformer_sign_model import TransformerSignModel
from LSC50Dataset import LSC50Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report
import numpy as np
from collections import Counter
import torch_optimizer as optim

def main():
    BASE_DIR = "Datasets/LANDMARKS"
    SEQ_LEN = 75
    BATCH_SIZE = 32
    EPOCHS = 20
    VAL_SPLIT = 0.2
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No se detectó GPU. Asegúrate de tener CUDA disponible.")
    DEVICE = torch.device("cuda")

    print(f"🚀 Dispositivo: {DEVICE}")
    dataset = LSC50Dataset(BASE_DIR, seq_len=SEQ_LEN, use_body=False, use_face=False, use_hands=True)
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = Counter(labels)
    print("📊 Ejemplos por clase:", class_counts)

    example_X, _ = dataset[0]
    INPUT_DIM = example_X.shape[1]
    print(f"🧠 Número total de características por frame (input_dim): {INPUT_DIM}")

    total_size = len(dataset)
    val_size = int(VAL_SPLIT * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Balanceo: calcular pesos por clase en train_dataset
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_sample_count = np.array([train_labels.count(c) for c in range(50)])
    weights_per_class = 1. / (class_sample_count + 1e-6)
    sample_weights = [weights_per_class[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = TransformerSignModel(INPUT_DIM, seq_len=SEQ_LEN, dim_model=256).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Lookahead(optim.RAdam(model.parameters(), lr=1e-4))
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        acc = np.mean(np.array(val_preds) == np.array(val_labels))
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {running_loss/train_size:.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "modelo_lsc_transformer.pth")
            print("✅ Mejor modelo guardado")

    print("📊 Reporte de clasificación:")
    print(classification_report(val_labels, val_preds, digits=4))

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    print("\n▶️ Iniciando entrenamiento...\n")
    print(torch.cuda.device_count())
    main()
