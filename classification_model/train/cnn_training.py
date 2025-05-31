import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torchvision.transforms import RandomChoice, RandomResizedCrop, ColorJitter

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

class CustomImageDataset(Dataset):
    def __init__(self, filepaths, transform_map=None, default_transform=None):
        self.filepaths = filepaths
        self.transform_map = transform_map or {}
        self.default_transform = default_transform
        self.class_names = sorted(list({os.path.basename(os.path.dirname(fp)) for fp in filepaths}))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label_name = os.path.basename(os.path.dirname(fp))
        label = self.class_to_idx[label_name]
        img = Image.open(fp).convert("RGB")
        transform = self.transform_map.get(label, self.default_transform)
        if transform:
            img = transform(img)
        return img, label


def cnn_dataloaders(train_files, val_files, test_files,
                    batch_size=8, img_size=224,
                    threshold=2500,
                    num_workers=0):

    basic_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    heavy_transform = transforms.Compose([
        RandomChoice([
            RandomResizedCrop(size=img_size, scale=(0.25, 0.25), ratio=(1.0, 1.0)),
            RandomResizedCrop(size=img_size, scale=(0.0625, 0.0625), ratio=(1.0, 1.0)),
        ]),
        transforms.RandomHorizontalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
    ])

    temp_ds = CustomImageDataset(train_files)
    idx_map = temp_ds.class_to_idx

    class_to_files = {idx: [] for idx in idx_map.values()}
    for fp in train_files:
        label = idx_map[os.path.basename(os.path.dirname(fp))]
        class_to_files[label].append(fp)

    balanced_train = []
    heavy_idx = set()
    for label, files in class_to_files.items():
        n = len(files)
        if n < threshold:
            heavy_idx.add(label)

            sampled = random.choices(files, k=threshold)
        else:

            sampled = random.sample(files, threshold)
        balanced_train.extend(sampled)

    transform_map = {i: heavy_transform for i in heavy_idx}
    default_transform = basic_transform

    train_ds = CustomImageDataset(balanced_train, transform_map=transform_map, default_transform=default_transform)
    val_ds   = CustomImageDataset(val_files,   transform_map={}, default_transform=basic_transform)
    test_ds  = CustomImageDataset(test_files,  transform_map={}, default_transform=basic_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, temp_ds.class_names


def create_cnn(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += imgs.size(0)
    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += imgs.size(0)
    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

def train_cnn(train_loader, val_loader, class_names,
              lr=1e-3, epochs=10, device="cpu", save_path="best_cnn.pth"):
    num_classes = len(class_names)
    model = create_cnn(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_acc_history = []

    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = eval_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Best model saved with val_acc={best_acc:.4f}")

    model.load_state_dict(torch.load(save_path))
    return model, train_loss_history, val_acc_history

def test_cnn(model, test_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"[Test] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    return test_loss, test_acc
