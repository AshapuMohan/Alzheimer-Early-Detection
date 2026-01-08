# ======================
# GOOGLE COLAB SETUP
# ======================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
# from google.colab import drive

# ======================
# MOUNT DRIVE (OPTIONAL)
# ======================
# drive.mount('/content/drive')

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATASET_PATH = "/content/Data/Data"   # change if using Drive
MODEL_PATH = "/content/drive/MyDrive/resnet18_alzheimer_best.pth"

NUM_CLASSES = 4
BATCH_SIZE = 32        # T4 GPU safe
EPOCHS = 15
LR = 1e-3

# ======================
# TRANSFORMS (MRI-SAFE)
# ======================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

# ======================
# LOAD DATASET
# ======================
full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)

class_names = full_dataset.classes
print("Classes:", class_names)
print("Total images:", len(full_dataset))

# ======================
# TRAIN / VAL SPLIT
# ======================
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size,
     val_size],
    generator=torch.Generator().manual_seed(42)
)

val_dataset.dataset.transform = val_transform

# ======================
# HANDLE CLASS IMBALANCE
# ======================
targets = [label for _, label in train_dataset]
class_counts = Counter(targets)

class_weights = {
    cls: 1.0 / count for cls, count in class_counts.items()
}

sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ======================
# DATALOADERS
# ======================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ======================
# MODEL – RESNET18
# ======================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ======================
# LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.fc.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

# ======================
# TRAINING LOOP
# ======================
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    scheduler.step(acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} | Val Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Best model saved")

# ======================
# FINAL EVALUATION
# ======================
print("\n🏆 BEST VALIDATION ACCURACY:", best_acc)

print("\n📊 Classification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\n📁 Model saved at:", MODEL_PATH)
