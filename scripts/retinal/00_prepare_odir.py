"""Phase 0 — organize ODIR-5K color-fundus images into the LAD class-folder layout.

ODIR-5K ships as one flat ``preprocessed_images/`` pool plus a ``full_df.csv`` with
per-eye single-label flags (columns N,D,G,C,A,H,M,O). This script:

  1. keeps single-label rows whose disease is in ``--classes`` and whose image is on disk,
  2. does a stratified train/val split (seeded),
  3. materializes ``RETINAL_ROOT/images/<class>/<filename>`` (hard-link on the same NTFS
     volume, copy fallback) — the raw pool that feeds classifier training (Phase 1) and
     step-02 correct-classification filtering (Phase 2),
  4. writes ``odir_manifest.csv`` (filename, class, class_idx, split) and ``classes.txt``
     (the canonical class order — the source of truth for the classifier head and the
     concept-map class indices, analogous to imagenet1k_classes.csv).

The ODIR *source* archive lives outside the repo and is passed via ``--odir-src`` (or
``$LAD_ODIR_SRC``); nothing absolute is hard-coded here.

Usage:
  python scripts/retinal/00_prepare_odir.py \
    --odir-src /path/to/ODIR-5K/archive
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# Make repo root importable so we can use paths.py
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

# ODIR's eight single-label flag columns -> canonical LAD class names.
FLAG_TO_CLASS = {
    "N": "normal",
    "D": "diabetic_retinopathy",
    "G": "glaucoma",
    "C": "cataract",
    "A": "amd",
    "H": "hypertension",
    "M": "myopia",
    "O": "other",
}
ALL_FLAGS = list(FLAG_TO_CLASS.keys())

# The "clean" reduced set: well-populated, visually distinct ocular diseases.
DEFAULT_CLASSES = ["normal", "diabetic_retinopathy", "glaucoma", "cataract", "amd"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--odir-src", default=os.environ.get("LAD_ODIR_SRC"),
                   help="ODIR archive dir containing full_df.csv and preprocessed_images/. "
                        "Defaults to $LAD_ODIR_SRC.")
    p.add_argument("--out", type=Path, default=paths.RETINAL_ROOT,
                   help=f"Output root for the organized pool (default: {paths.RETINAL_ROOT}).")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   help=f"Class names to keep, in canonical order (default: {DEFAULT_CLASSES}).")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Per-class validation fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["link", "copy"], default="link",
                   help="'link' = hard-link (instant, same volume); 'copy' = duplicate bytes.")
    p.add_argument("--limit-per-class", type=int, default=None,
                   help="Optional cap per class (for a fast laptop smoke run).")
    return p.parse_args()


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "link":
        try:
            os.link(src, dst)  # hard link: free on the same NTFS volume, no admin needed
            return
        except OSError:
            pass  # cross-volume or unsupported -> fall back to copy
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    if not args.odir_src:
        raise SystemExit("Pass --odir-src (or set $LAD_ODIR_SRC) to the ODIR archive directory.")
    src_root = Path(args.odir_src)
    csv_path = src_root / "full_df.csv"
    img_root = src_root / "preprocessed_images"
    for pth, what in [(csv_path, "full_df.csv"), (img_root, "preprocessed_images/")]:
        if not pth.exists():
            raise SystemExit(f"Expected {what} at {pth} — check --odir-src.")

    class_to_idx = {c: i for i, c in enumerate(args.classes)}

    df = pd.read_csv(csv_path)
    df["nflag"] = df[ALL_FLAGS].sum(axis=1)
    single = df[df["nflag"] == 1].copy()

    def row_class(r):
        for f in ALL_FLAGS:
            if r[f] == 1:
                return FLAG_TO_CLASS[f]
        return None

    single["cls"] = single.apply(row_class, axis=1)
    single = single[single["cls"].isin(class_to_idx)]
    single["src_path"] = single["filename"].apply(lambda fn: img_root / str(fn))
    single = single[single["src_path"].apply(lambda p: p.exists())].reset_index(drop=True)

    out_root = Path(args.out)
    images_dir = out_root / "images"
    rows = []
    print(f"ODIR source: {src_root}")
    print(f"Output root: {out_root}\n")
    print(f"{'class':<22}{'kept':>7}{'train':>7}{'val':>7}")
    print("-" * 43)

    rng_cols = ["filename", "cls", "src_path"]
    for cls in args.classes:
        sub = single[single["cls"] == cls][rng_cols].sample(frac=1.0, random_state=args.seed)
        if args.limit_per_class:
            sub = sub.iloc[: args.limit_per_class]
        n = len(sub)
        n_val = int(round(n * args.val_ratio))
        for i, (_, r) in enumerate(sub.iterrows()):
            split = "val" if i < n_val else "train"
            dst = images_dir / cls / str(r["filename"])
            materialize(Path(r["src_path"]), dst, args.mode)
            rows.append({
                "filename": r["filename"],
                "class": cls,
                "class_idx": class_to_idx[cls],
                "split": split,
                "path": str(dst.relative_to(out_root)).replace("\\", "/"),
            })
        print(f"{cls:<22}{n:>7}{n - n_val:>7}{n_val:>7}")

    manifest = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_root / "odir_manifest.csv", index=False)
    (out_root / "classes.txt").write_text("\n".join(args.classes) + "\n", encoding="utf-8")

    print("-" * 43)
    print(f"{'TOTAL':<22}{len(manifest):>7}"
          f"{int((manifest.split == 'train').sum()):>7}"
          f"{int((manifest.split == 'val').sum()):>7}")
    print(f"\nWrote {out_root/'odir_manifest.csv'}")
    print(f"Wrote {out_root/'classes.txt'}  (class order = head index order)")
    print(f"Images under {images_dir}/<class>/  (mode={args.mode})")


if __name__ == "__main__":
    main()
