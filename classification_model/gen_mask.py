from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from cellpose import models


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="root of raw images")
    p.add_argument("--dst", required=True, help="root to save masked images")
    p.add_argument("--ext", default="png", help="image extension (e.g. png, jpg)")
    p.add_argument("--diam", type=float, default=None, help="nucleus diameter px")
    p.add_argument("--gpu", action="store_true", help="use GPU for Cellpose")
    p.add_argument(
        "--batch",
        type=int,
        default=64,
        help="number of images per GPU batch (increase until you hit GPU OOM)",
    )
    return p.parse_args()


def is_valid_path(path: Path) -> bool:
    return "__MACOSX" not in path.parts and not path.name.startswith("._")


def load_image(path: Path):
    try:
        return np.array(Image.open(path))
    except (UnidentifiedImageError, OSError):
        return None


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)

    paths = [p for p in sorted(src.rglob(f"*.{args.ext}")) if is_valid_path(p)]
    if not paths:
        sys.exit("No valid images found!")

    dst.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(gpu=args.gpu)

    skipped = 0
    for chunk in tqdm(list(batched(paths, args.batch)), desc="Masking", unit="batch"):
        imgs, valid_paths = [], []
        for p in chunk:
            arr = load_image(p)
            if arr is not None:
                imgs.append(arr)
                valid_paths.append(p)
            else:
                skipped += 1
                tqdm.write(f"Skipped invalid image: {p}")

        if not imgs:
            continue

        masks_list, _, _ = model.eval(
            imgs, channel_axis=-1, diameter=args.diam, batch_size=len(imgs)
        )

        for p, rgb, masks in zip(valid_paths, imgs, masks_list):
            if masks is None or masks.size == 0:
                rgb[:] = 0
            else:
                rgb[masks == 0] = 0

            out = dst / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(out, optimize=True)

    if skipped:
        print(f"Finished with {skipped} resource‑fork/invalid files skipped.")


if __name__ == "__main__":
    main()
