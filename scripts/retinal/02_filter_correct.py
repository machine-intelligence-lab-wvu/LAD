"""Phase 2 — keep only fundus images the trained classifier gets right.

Concept maps explain *correct* model decisions, so (as in the ImageNet/CXR pipelines) we
forward-pass the Phase-1 classifier over the organized pool and materialize the correctly-
classified images into ``FILTERED_ROOT/<backbone>/correct/<class>/`` (hard-links). Class
indices come from the checkpoint's own ``classes`` list — no ImageNet label space involved.

Usage (lad conda env):
  python scripts/retinal/02_filter_correct.py --backbone resnet34
  python scripts/retinal/02_filter_correct.py --backbone resnet34 --split val
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402
from lad.backbones import load_classifier_checkpoint  # noqa: E402
from lad.data import imagenet_transform  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--checkpoint", type=Path, default=None, help="default: <MODELS_DIR>/best_<backbone>_odir.pt")
    p.add_argument("--data-root", type=Path, default=paths.RETINAL_ROOT)
    p.add_argument("--out", type=Path, default=None, help="default: <RETINAL_FILTERED_ROOT>")
    p.add_argument("--split", choices=["all", "train", "val"], default="all")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mode", choices=["link", "copy"], default="link")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class _ImgDS(Dataset):
    def __init__(self, rows, data_root, tf):
        self.rows, self.root, self.tf = rows, Path(data_root), tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(self.root / r["path"]).convert("RGB")), i


def materialize(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    ckpt = args.checkpoint or (paths.MODELS_DIR / f"best_{args.backbone}_odir.pt")
    out_root = (args.out or paths.RETINAL_FILTERED_ROOT) / args.backbone / "correct"

    model, backbone, classes = load_classifier_checkpoint(ckpt, device=args.device)
    print(f"loaded {ckpt}  backbone={backbone}  classes={classes}")

    df = pd.read_csv(args.data_root / "odir_manifest.csv")
    if args.split != "all":
        df = df[df.split == args.split]
    rows = df.to_dict("records")
    ds = _ImgDS(rows, args.data_root, imagenet_transform())
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))

    kept = {c: 0 for c in classes}
    total = {c: 0 for c in classes}
    with torch.inference_mode():
        for x, idxs in ld:
            preds = model(x.to(args.device)).argmax(1).cpu().tolist()
            for pi, di in zip(preds, idxs.tolist()):
                r = rows[di]
                cls = r["class"]
                total[cls] += 1
                if pi == int(r["class_idx"]):
                    kept[cls] += 1
                    src = args.data_root / r["path"]
                    materialize(src, out_root / cls / Path(r["path"]).name, args.mode)

    print(f"\n{'class':<22}{'kept':>7}{'total':>7}{'acc':>8}")
    print("-" * 44)
    for c in classes:
        acc = kept[c] / max(total[c], 1)
        print(f"{c:<22}{kept[c]:>7}{total[c]:>7}{acc:>8.3f}")
    K, T = sum(kept.values()), sum(total.values())
    print("-" * 44)
    print(f"{'TOTAL':<22}{K:>7}{T:>7}{K / max(T, 1):>8.3f}")
    print(f"\nCorrect images -> {out_root}/<class>/  (mode={args.mode}, split={args.split})")


if __name__ == "__main__":
    main()
