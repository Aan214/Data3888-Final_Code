import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return img


def image_to_patches(img_pil, patch_size=32):

    w, h = img_pil.size
    patches = []
    n_x = w // patch_size
    n_y = h // patch_size

    for iy in range(n_y):
        for ix in range(n_x):
            left = ix * patch_size
            top = iy * patch_size
            right = left + patch_size
            bottom = top + patch_size
            patch = img_pil.crop((left, top, right, bottom))
            patches.append(patch)

    return patches, w, h, n_x, n_y


def extract_features(patches):

    resnet = torchvision.models.resnet50(pretrained=True)

    resnet.fc = nn.Identity()
    resnet.eval()
    resnet.to(DEVICE)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),

        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    all_features = []
    batch_pil = []
    for p in patches:
        batch_pil.append(p)

        if len(batch_pil) == BATCH_SIZE:
            feats = forward_batch(batch_pil, transform, resnet)
            all_features.append(feats)
            batch_pil = []

    if len(batch_pil) > 0:
        feats = forward_batch(batch_pil, transform, resnet)
        all_features.append(feats)

    all_features = torch.cat(all_features, dim=0)
    return all_features.cpu().numpy()


def forward_batch(pil_list, transform, model):

    imgs = [transform(p) for p in pil_list]

    imgs_tensor = torch.stack(imgs, dim=0).to(DEVICE)
    with torch.no_grad():
        feats = model(imgs_tensor)
    return feats


def cluster_features(feat_array, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(feat_array)
    return labels


def labels_to_seg_image(labels, n_x, n_y, patch_size=32):
    label_map = labels.reshape(n_y, n_x)

    max_label = label_map.max()
    if max_label == 0:
        max_label = 1
    normalized = (label_map / max_label) * 255.0
    normalized = normalized.astype(np.uint8)

    seg_h = n_y * patch_size
    seg_w = n_x * patch_size
    seg_arr = np.zeros((seg_h, seg_w), dtype=np.uint8)
    for iy in range(n_y):
        for ix in range(n_x):
            v = normalized[iy, ix]

            top = iy * patch_size
            left = ix * patch_size
            seg_arr[top:top + patch_size, left:left + patch_size] = v

    seg_map_pil = Image.fromarray(seg_arr, mode='L')
    return seg_map_pil


def main_demo(img_path, patch_size=32, n_clusters=2, out_dir="results"):

    os.makedirs(out_dir, exist_ok=True)

    img_pil = load_image(img_path)
    patches, w, h, n_x, n_y = image_to_patches(img_pil, patch_size)

    feats = extract_features(patches)

    labels = cluster_features(feats, n_clusters=n_clusters)

    seg_map = labels_to_seg_image(labels, n_x, n_y, patch_size)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    seg_save_path = os.path.join(out_dir, f"{base_name}_seg_k{n_clusters}.png")
    seg_map.save(seg_save_path)
    print(f"Segmentation map saved to: {seg_save_path}")


if __name__ == "__main__":
    test_image = "example_cell.png"
    main_demo(test_image, patch_size=1, n_clusters=2)
