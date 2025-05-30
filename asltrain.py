import kagglehub
import torch
import torchvision.models as models
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# Transformaciones básicas
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset desde carpeta
dataset_path = f"{path}/asl_alphabet_train"  # cambia si lo renombraste
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print(f"✅ Clases: {dataset.classes}")


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))  # 29 clases
