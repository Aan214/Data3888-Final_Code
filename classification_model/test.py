import torch
from train.swin_training import (
    swin_dataloaders,
    create_swin,
    test_swin
)

from train.cnn_training import (
    cnn_dataloaders,
    create_cnn,
    test_cnn
)
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from multiprocessing import freeze_support
import numpy as np
from collections import defaultdict


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

        raw_class_names = set()
        for f in file_list:
            class_name = os.path.basename(os.path.dirname(f))
            if class_name.startswith('DCIS'):
                class_name = 'DCIS'
            raw_class_names.add(class_name)

        self.class_names = sorted(list(raw_class_names))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        self.class_counts = defaultdict(int)
        for f in file_list:
            class_name = os.path.basename(os.path.dirname(f))
            if class_name.startswith('DCIS'):
                class_name = 'DCIS'
            self.class_counts[class_name] += 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_name = os.path.basename(os.path.dirname(img_path))
        if label_name.startswith('DCIS'):
            label_name = 'DCIS'
        label = self.class_to_idx[label_name]
        return image, label


def evaluate_metrics(model, dataloader, device, class_names):
    n_cls = len(class_names)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    counts = np.zeros((n_cls, n_cls), dtype=int)
    running_loss = 0.0
    total_samples = 0
    correct_top3 = 0

    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            batch_size = imgs.size(0)
            total_samples += batch_size

            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                counts[t, p] += 1

            top3 = outputs.topk(3, dim=1).indices
            correct_top3 += (
                top3 == labels.view(-1, 1)
            ).any(dim=1).sum().item()

    overall_acc = np.trace(counts) / total_samples
    overall_loss = running_loss / total_samples
    top3_acc = correct_top3 / total_samples

    per_class_acc = {}
    mis_top3 = {}
    for idx, cls in enumerate(class_names):
        total = counts[idx].sum()
        correct = counts[idx, idx]
        per_class_acc[cls] = correct / total if total else 0.0

        wrong = counts[idx].copy()
        wrong[idx] = 0
        if wrong.sum() == 0:
            mis_top3[cls] = []
        else:
            top3_idx = wrong.argsort()[::-1][:3]
            mis_top3[cls] = [
                (class_names[j], int(wrong[j]), wrong[j] / total)
                for j in top3_idx if wrong[j] > 0
            ]

    return overall_loss, overall_acc, top3_acc, per_class_acc, mis_top3, counts


def main():
    with open("dataset_processed/test.txt", "r") as f:
        test_files = f.read().strip().splitlines()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(test_files, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    class_names = test_dataset.class_names

    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing: {device}")

    num_classes = len(class_names)
    model = create_swin(num_classes, model_name="swin_base_patch4_window7_224", pretrained=False)
    model.load_state_dict(torch.load("saved_model/best_swin.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_loss, test_acc = test_swin(model, test_loader, device)
    print(f"\n[test_swin] Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

    loss, acc, top3_acc, per_acc, mis_top3, counts = evaluate_metrics(
        model, test_loader, device, class_names
    )

    print("\nDetailed evaluation:")
    print(f"Loss            : {loss:.4f}")
    print(f"Top‑1 Accuracy  : {acc :.4f}")
    print(f"Top‑3 Accuracy  : {top3_acc:.4f}")

    print("\nPer‑class accuracy:")
    for cls, a in per_acc.items():
        print(f"{cls:<15}: {a:.4f}")

    print("\nTop‑3 mis‑classification for each class:")
    for cls, lst in mis_top3.items():
        if not lst:
            print(f"{cls:<15}: (no errors)")
        else:
            pretty = ",  ".join([
                f"{pred} ({n} / {ratio:.2%})" for pred, n, ratio in lst
            ])
            print(f"{cls:<15}: {pretty}")

    pos_idx = [
        i for i, name in enumerate(class_names)
        if ("DCIS" in name) or ("Invasive_Tumor" in name)
    ]
    neg_idx = [i for i in range(len(class_names)) if i not in pos_idx]

    TP = counts[np.ix_(pos_idx, pos_idx)].sum()
    FN = counts[np.ix_(pos_idx, neg_idx)].sum()
    FP = counts[np.ix_(neg_idx, pos_idx)].sum()
    TN = counts[np.ix_(neg_idx, neg_idx)].sum()

    print("\nBinary confusion matrix (Positive=IDC+DCIS, Negative=others)")
    print(f"            Pred_Pos  Pred_Neg")
    print(f"True_Pos    {TP:9d} {FN:9d}")
    print(f"True_Neg    {FP:9d} {TN:9d}")

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\nPrecision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")


if __name__ == '__main__':
    freeze_support()
    main()