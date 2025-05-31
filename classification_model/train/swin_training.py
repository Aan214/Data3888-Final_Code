import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random

import torch
import timm
import math


class CustomImageDataset(Dataset):
    def __init__(self,
                 filepaths,
                 transform=None,
                 augment_transform=None,
                 upper_limit=None,
                 lower_limit=None):

        raw_class_names = set()
        for fp in filepaths:
            cls = os.path.basename(os.path.dirname(fp))
            if cls.startswith("DCIS"):
                cls = "DCIS"
            raw_class_names.add(cls)
        self.class_names = sorted(list(raw_class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        if upper_limit is not None and len(filepaths) > upper_limit:
            filepaths = random.sample(filepaths, upper_limit)

        self.original_len = len(filepaths)
        self.transform = transform
        self.augment_transform = augment_transform or transform

        if lower_limit is not None and len(filepaths) < lower_limit:
            repeats = math.ceil(lower_limit / len(filepaths))
            filepaths = filepaths * repeats
            filepaths = filepaths[:lower_limit]
            self.has_extras = True
        else:
            self.has_extras = False

        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label_name = os.path.basename(os.path.dirname(fp))
        if label_name.startswith("DCIS"):
            label_name = "DCIS"
        label = self.class_to_idx[label_name]

        img = Image.open(fp).convert("RGB")

        if self.has_extras and idx >= self.original_len:
            img = self.augment_transform(img)
        else:
            img = self.transform(img)

        return img, label


def swin_dataloaders(train_files, val_files, test_files,
                     batch_size=64, img_size=224, num_workers=0,
                     upper_limit=None, lower_limit=None):
    img_tf_resize = transforms.Resize((img_size, img_size))

    train_transform = transforms.Compose([
        img_tf_resize,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        img_tf_resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    stronger_augment = transforms.Compose([
        img_tf_resize,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CustomImageDataset(
        train_files,
        transform=train_transform,
        augment_transform=stronger_augment,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
    )

    val_ds = CustomImageDataset(val_files, transform=val_transform)
    test_ds = CustomImageDataset(test_files, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.class_names



def create_swin(num_classes, model_name="swin_base_patch4_window7_224", pretrained=True):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += imgs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = float(running_corrects) / total_samples
    return epoch_loss, epoch_acc


def eval_model(model, dataloader, criterion, device="cpu"):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += imgs.size(0)

    epoch_loss = running_loss / total_samples if total_samples>0 else 0
    epoch_acc = float(running_corrects) / total_samples if total_samples>0 else 0
    return epoch_loss, epoch_acc


def train_swin(train_loader, val_loader, class_names,
               model_name="swin_base_patch4_window7_224",
               pretrained=True,
               lr=1e-4,
               num_epochs=10,
               device="cpu",
               save_path="best_swin.pth"):

    num_classes = len(class_names)
    model = create_swin(num_classes, model_name=model_name, pretrained=pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loss_history = []
    val_acc_history = []

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = eval_model(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Best model saved (val_acc={best_val_acc:.4f})")

    model.load_state_dict(torch.load(save_path))
    return model, train_loss_history, val_acc_history


def test_swin(model, test_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"[Test] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    return test_loss, test_acc
