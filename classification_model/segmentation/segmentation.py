import os
import csv
import random
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def generate_multiclass_masks(class_dirs,
                              mask_root,
                              mapping_csv,
                              n_clusters=2,
                              class_thresholds=None,
                              default_threshold=0.3):
    class_thresholds = class_thresholds or {}
    os.makedirs(mask_root, exist_ok=True)
    palette = {0:(0,0,0),1:(255,0,0),2:(0,255,0),3:(0,0,255)}

    with open(mapping_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path','mask_path'])
        for cls_idx,(cls_name,cls_dir) in enumerate(class_dirs, start=1):
            max_ratio = class_thresholds.get(cls_name, default_threshold)
            for fname in tqdm(os.listdir(cls_dir), desc=f"Label {cls_name}"):
                if not fname.lower().endswith('.png'): continue
                img_p = os.path.join(cls_dir, fname)
                gray = Image.open(img_p).convert('L')
                arr  = np.array(gray).reshape(-1,1)
                km   = KMeans(n_clusters=n_clusters, random_state=0)
                labels = km.fit_predict(arr)
                means = [arr[labels==i].mean() for i in range(n_clusters)]
                fg    = int(np.argmin(means))
                bin_m = (labels.reshape(gray.size[::-1])==fg).astype(np.uint8)
                frac  = bin_m.mean()
                if not (0<frac<=max_ratio): continue
                mask_multi = (bin_m*cls_idx).astype(np.uint8)
                out_dir = os.path.join(mask_root, cls_name)
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(fname)[0]
                raw_p   = os.path.join(out_dir, base+'_mask.png')
                color_p = os.path.join(out_dir, base+'_mask_color.png')
                Image.fromarray(mask_multi).save(raw_p)
                writer.writerow([img_p, raw_p])

                h,w = mask_multi.shape
                rgb = np.zeros((h,w,3),dtype=np.uint8)
                for c,col in palette.items():
                    rgb[mask_multi==c] = col
                Image.fromarray(rgb).save(color_p)
    print(f"Multi-class masks generated, mapping → {mapping_csv}")

def compute_cell_metrics_batch(preds, gts, num_classes):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(gts,   torch.Tensor):
        gts   = gts.cpu().numpy()
    preds = preds.reshape(-1)
    gts   = gts.reshape(-1)
    class_iou  = np.zeros(num_classes)
    class_dice = np.zeros(num_classes)
    for c in range(num_classes):
        p_c = preds==c; g_c = gts==c
        inter = np.logical_and(p_c,g_c).sum()
        union = np.logical_or(p_c,g_c).sum()
        class_iou[c]  = inter/union if union>0 else np.nan
        sum_sz = p_c.sum()+g_c.sum()
        class_dice[c] = 2*inter/sum_sz if sum_sz>0 else np.nan
    fg = gts>=1
    cell_acc = (preds[fg]==gts[fg]).sum()/fg.sum() if fg.sum()>0 else np.nan
    return cell_acc, class_iou, class_dice

class SegDataset(Dataset):
    def __init__(self, mappings, img_transform=None):
        self.items = mappings
        self.img_transform = img_transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        img_p, mask_p = self.items[idx]
        img  = Image.open(img_p).convert('RGB')
        mask = Image.open(mask_p)
        if self.img_transform:
            img  = self.img_transform(img)
            mask = transforms.Resize((256,256), interpolation=Image.NEAREST)(mask)
        mask_arr = np.array(mask, dtype=np.int64)
        return img, torch.from_numpy(mask_arr)

def train_segmentation(mapping_csv, model_save,
                       num_classes=4,
                       epochs=20, batch_size=8, lr=1e-3,
                       num_workers=8, device=None):

    items = []
    with open(mapping_csv) as f:
        for row in csv.DictReader(f):
            items.append((row['image_path'], row['mask_path']))
    random.shuffle(items)
    if not items:
        raise RuntimeError("No entries in mapping_csv")

    class_to_items = defaultdict(list)
    for img_p, mask_p in items:
        cls = os.path.basename(os.path.dirname(mask_p))
        class_to_items[cls].append((img_p,mask_p))
    min_n = min(len(v) for v in class_to_items.values())
    balanced = []
    for v in class_to_items.values():
        balanced.extend(random.sample(v, min_n))
    random.shuffle(balanced)
    items = balanced
    print(f"Balanced train: {len(items)} samples ({min_n}/class)")

    img_tr = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    ds = SegDataset(items, img_transform=img_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=False, pretrained_backbone=True, num_classes=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss, train_acc = [], []
    for ep in range(1, epochs+1):
        total_loss=0; total_corr=0; total_pix=0
        epoch_preds, epoch_gts = [], []
        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        model.train()
        for imgs,masks in pbar:
            imgs,masks = imgs.to(device), masks.to(device)
            out = model(imgs)['out']
            loss = criterion(out, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            b = imgs.size(0)
            total_loss += loss.item()*b
            preds = out.argmax(1)
            total_corr += (preds==masks).sum().item()
            total_pix  += preds.numel()
            epoch_preds.append(preds.cpu().numpy())
            epoch_gts.append(masks.cpu().numpy())
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss/len(ds)
        avg_acc  = total_corr/total_pix
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)

        all_p = np.concatenate(epoch_preds,0)
        all_g = np.concatenate(epoch_gts,0)
        cell_acc, cls_iou, cls_dice = compute_cell_metrics_batch(all_p, all_g, num_classes)
        mIoU  = np.nanmean(cls_iou[1:])
        mDice = np.nanmean(cls_dice[1:])
        print(f"Epoch {ep}/{epochs} Loss={avg_loss:.4f} "
              f"OverallAcc={avg_acc:.4f} CellAcc={cell_acc:.4f} "
              f"mIoU={mIoU:.4f} mDice={mDice:.4f}")
        for c in range(1, num_classes):
            print(f"   Class{c}: IoU={cls_iou[c]:.4f} Dice={cls_dice[c]:.4f}")

    torch.save(model.state_dict(), model_save)
    print(f"Model saved to {model_save}")

    # plot
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_loss, marker='o')
    plt.title("Train Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_acc, marker='o')
    plt.title("Train Overall Acc"); plt.xlabel("Epoch"); plt.ylabel("Acc")
    plt.tight_layout(); plt.savefig("train_metrics.png"); plt.show()

    return model

def visualize_multiclass(model, datasets, output_dir, class_colors, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    tr = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    model.eval()
    for ds in datasets:
        od = os.path.join(output_dir, os.path.basename(ds))
        os.makedirs(od, exist_ok=True)
        for fn in os.listdir(ds):
            if not fn.lower().endswith('.png'): continue
            img = Image.open(os.path.join(ds,fn)).convert('RGB')
            inp = tr(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp)['out'].argmax(1).cpu().squeeze(0).numpy()
            ov = np.array(img.resize((256,256))).copy()
            for cls,col in class_colors.items():
                ov[pred==cls] = col
            Image.fromarray(ov).save(os.path.join(od,fn))
    print(f"Visualization saved to '{output_dir}'")

if __name__=='__main__':
    class_dirs = [
        ('Stromal','../dataset_raw/100/Stromal/Stromal'),
        ('CD4+_T_Cells','../dataset_raw/100/CD4+_T_Cells/CD4+_T_Cells'),
        ('CD8+_T_Cells','../dataset_raw/100/CD8+_T_Cells/CD8+_T_Cells'),
        ('Invasive_Tumor','../dataset_raw/100/Invasive_Tumor/Invasive_Tumor'),
    ]
    class_thresholds = {'Stromal':0.2,'CD4+_T_Cells':0.2,'CD8+_T_Cells':0.2,'Invasive_Tumor':0.4}
    seg_model = train_segmentation(
        mapping_csv='mapping_multiclass.csv',
        model_save='seg_deeplabv3.pth',
        num_classes=5,
        epochs=200,
        batch_size=8,
        lr=1e-3,
        num_workers=22
    )
    class_colors = {1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255,0,255)}
    visualize_multiclass(
        model=seg_model,
        datasets=[
            '../dataset_raw/100/Stromal_and_T_Cell_Hybrid/Stromal_and_T_Cell_Hybrid',
            '../dataset_raw/100/T_Cell_and_Tumor_Hybrid/T_Cell_and_Tumor_Hybrid'
        ],
        output_dir='vis_deeplabv3',
        class_colors=class_colors
    )
