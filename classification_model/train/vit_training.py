import os
import random
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


class CustomImageDataset(Dataset):
    def __init__(
        self,
        filepaths,
        transform=None,
        augment_transform=None,
        upper_limit=None,
        lower_limit=None
    ):

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


def vit_dataloaders(
    train_files, val_files, test_files,
    batch_size=8, img_size=224, num_workers=0,
    upper_limit=None, lower_limit=None
):
    img_tf_resize = transforms.Resize((img_size, img_size))

    train_transform = transforms.Compose([
        img_tf_resize,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        img_tf_resize,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    stronger_augment = train_transform

    train_ds = CustomImageDataset(
        train_files,
        transform=train_transform,
        augment_transform=stronger_augment,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
    )
    val_ds  = CustomImageDataset(val_files,  transform=val_transform)
    test_ds = CustomImageDataset(test_files, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.class_names


def create_vit(num_classes, model_name="vit_base_patch16_224", pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss      += loss.item() * imgs.size(0)
        running_corrects  += torch.sum(preds == labels).item()
        total             += imgs.size(0)

    return running_loss / total, running_corrects / total


@torch.no_grad()
def eval_model(model, dataloader, criterion, device="cpu"):
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss     += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total            += imgs.size(0)

    return (running_loss / total if total else 0,
            running_corrects / total if total else 0)


def train_vit(
    train_loader, val_loader, class_names,
    model_name="vit_base_patch16_224",
    pretrained=True, lr=1e-4, num_epochs=10,
    device="cpu", save_path="best_vit.pth"
):
    model = create_vit(len(class_names), model_name, pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0
    train_loss_hist, val_acc_hist = [], []

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(f"[{epoch+1:02d}/{num_epochs}] "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
              f"val loss {val_loss:.4f}  acc {val_acc:.4f}")

        train_loss_hist.append(tr_loss)
        val_acc_hist.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [*] best model saved  (val_acc={best_val_acc:.4f})")

    model.load_state_dict(torch.load(save_path))
    return model, train_loss_hist, val_acc_hist


def test_vit(model, test_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    loss, acc = eval_model(model, test_loader, criterion, device)
    print(f"[Test] loss {loss:.4f}  acc {acc:.4f}")
    return loss, acc
