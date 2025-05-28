import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# === Descargar dataset ===
path = kagglehub.dataset_download("datamunge/sign-language-mnist")
train_path = f"{path}/sign_mnist_train.csv"
test_path = f"{path}/sign_mnist_test.csv"

# === Dataset personalizado ===
class SignMNISTDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].values
        self.images = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype('float32') / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx])
        return image, label

# === Modelo CNN simple ===
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=25):  # 25 letras (A-Y sin J)
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# === Cargar datos ===
train_dataset = SignMNISTDataset(train_path)
test_dataset = SignMNISTDataset(test_path)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# === Entrenamiento ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(num_classes=25).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss, total_correct = 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        total_correct += (out.argmax(1) == batch_y).sum().item()

    acc = total_correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader.dataset):.4f} - Accuracy: {acc:.2%}")

# === Evaluación ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))

# === Mostrar imágenes predichas (opcional) ===
import numpy as np
def show_sample():
    x, y = test_dataset[0]
    pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
    plt.imshow(x.squeeze(), cmap='gray')
    plt.title(f"Label: {chr(y + 65)}, Predicted: {chr(pred + 65)}")
    plt.show()
show_sample()

# === Guardar modelo entrenado ===
torch.save(model.state_dict(), "cnn_sign_mnist_model.pth")
print("✅ Modelo guardado como cnn_sign_mnist_model.pth")

