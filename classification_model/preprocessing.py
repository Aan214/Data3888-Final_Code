import os
import re
import random


def parse_scale(filename):
    m = re.match(r"cell_\d+_(\d+)\.png$", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def get_image_paths(split_dir, desired_scale=100, exclude_clusters=None):
    paths = {}
    for cluster in os.listdir(split_dir):
        if exclude_clusters and cluster in exclude_clusters:
            continue
        cluster_dir = os.path.join(split_dir, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        files = [f for f in os.listdir(cluster_dir) if f.lower().endswith('.png')]
        valid = []
        for f in files:
            scale = parse_scale(f)
            if scale == desired_scale:
                valid.append(os.path.join(cluster_dir, f))
        if valid:
            paths[cluster] = valid
    return paths


def balance_paths(paths_dict, seed=42):
    random.seed(seed)
    selected = []
    for cluster, imgs in paths_dict.items():
        print(f"Cluster '{cluster}': {len(imgs)} images")
        selected.extend(imgs)
    return selected


def write_list(filelist, out_txt):
    with open(out_txt, 'w') as f:
        for p in filelist:
            f.write(p + '\n')


if __name__ == "__main__":
    EXCLUDE_CLUSTERS = [
                        # "B_Cells",
                        # "CD4+_T_Cells",
                        # "CD8+_T_Cells",
                        # "DCIS_1",
                        # "DCIS_2",
                        # "Endothelial",
                        # "Invasive_Tumor",
                        "IRF7+_DCs",
                        "LAMP3+_DCs",
                        # "Macrophages_1",
                        # "Macrophages_2",
                        "Mast_Cells",
                        # "Myoepi_ACTA2+",
                        # "Myoepi_KRT15+",
                        "Perivascular-Like",
                        "Prolif_Invasive_Tumor",
                        # "Stromal",
                        "Stromal_and_T_Cell_Hybrid",
                        "T_Cell_and_Tumor_Hybrid",
                        "Unlabeled"]

    base_split = "dataset_splits"
    desired_scale = 100
    splits = ['train', 'validation', 'test']

    for split in splits:
        print(f"Processing split: {split}")
        split_dir = os.path.join(base_split, split)

        paths = get_image_paths(split_dir, desired_scale, exclude_clusters=EXCLUDE_CLUSTERS)

        selected = balance_paths(paths)

        out_txt = f"{split}.txt"
        write_list(selected, out_txt)
        print(f"  -> {out_txt} saved, {len(selected)} total samples.")