import os
import re
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

cbr_path = "./dataset_raw/metadata_code/cbr.csv"
base_dir = "./dataset_raw/100"
output_base = "./dataset_splits"

cbr = pd.read_csv(cbr_path)
bbox = (
    cbr
    .groupby('index')
    .agg(
        x_min0=('axis-1', 'min'),
        x_max0=('axis-1', 'max'),
        y_min0=('axis-0', 'min'),
        y_max0=('axis-0', 'max')
    )
)

records = []
pattern = re.compile(r'cell_(\d+)_(\d+)\.png$', re.IGNORECASE)
for ext in tqdm(os.listdir(base_dir), desc='Extensions'):
    ext_path = os.path.join(base_dir, ext)
    if not os.path.isdir(ext_path):
        continue
    for cluster in tqdm(os.listdir(ext_path), desc=f'Clusters ({ext})', leave=False):
        cluster_path = os.path.join(ext_path, cluster)
        if not os.path.isdir(cluster_path):
            continue
        for fname in tqdm(os.listdir(cluster_path), desc=f'Files ({ext}/{cluster})', leave=False):
            m = pattern.search(fname)
            if m:
                cell_id = int(m.group(1))
                ext_val = int(m.group(2))
                records.append({
                    'cluster': cluster,
                    'filename': fname,
                    'cell_id': cell_id,
                    'ext': ext_val,
                    'path': os.path.join(cluster_path, fname)
                })

df_files = pd.DataFrame.from_records(records)
if df_files.empty:
    raise RuntimeError("No files found. Please check your base_dir and filename patterns.")
df = (
    df_files
    .merge(bbox, left_on='cell_id', right_index=True)
    .assign(
        x_min=lambda d: np.floor(d['x_min0'] - d['ext']).astype(int),
        x_max=lambda d: np.ceil(d['x_max0'] + d['ext']).astype(int),
        y_min=lambda d: np.floor(d['y_min0'] - d['ext']).astype(int),
        y_max=lambda d: np.ceil(d['y_max0'] + d['ext']).astype(int)
    )
    .reset_index(drop=True)
)

df['group'] = 0
group_count = 0
for idx, row in tqdm(df.iterrows(), total=len(df), desc='Grouping', leave=False):
    existing_all = df[df['group'] > 0][['x_min', 'y_min', 'x_max', 'y_max', 'group']]
    placed = False
    for g in range(1, group_count + 1):
        existing = existing_all[existing_all['group'] == g]
        ex_x_min = existing['x_min'].to_numpy()
        ex_x_max = existing['x_max'].to_numpy()
        ex_y_min = existing['y_min'].to_numpy()
        ex_y_max = existing['y_max'].to_numpy()
        overlap = (
            (row['x_min'] < ex_x_max) &
            (row['x_max'] > ex_x_min) &
            (row['y_min'] < ex_y_max) &
            (row['y_max'] > ex_y_min)
        ).any()
        if not overlap:
            df.at[idx, 'group'] = g
            placed = True
            break
    if not placed:
        group_count += 1
        df.at[idx, 'group'] = group_count

os.makedirs(output_base, exist_ok=True)
group_df = df[['cluster', 'filename', 'cell_id', 'group']]
group_df.to_csv(os.path.join(output_base, 'group_assignments.csv'), index=False)
print(f"Group assignments saved to {os.path.join(output_base, 'group_assignments.csv')}")

groups = df['group'].unique()
train_groups, temp_groups = train_test_split(groups, test_size=0.3, random_state=42)
val_groups, test_groups = train_test_split(temp_groups, test_size=0.5, random_state=42)
df['set'] = df['group'].map(lambda g: 'train' if g in train_groups else ('validation' if g in val_groups else 'test'))

for set_name in ['train', 'validation', 'test']:
    for cluster in df['cluster'].unique():
        dir_path = os.path.join(output_base, set_name, cluster)
        os.makedirs(dir_path, exist_ok=True)
    subset = df[df['set'] == set_name]
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f'Copying {set_name}', leave=False):
        dest = os.path.join(output_base, set_name, row['cluster'], row['filename'])
        shutil.copy2(row['path'], dest)

print("Finished splitting and saving data to:", output_base)
