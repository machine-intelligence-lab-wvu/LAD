"""Batch concept-overlay visualization across many classes — backbone-agnostic.

For each class in --classes-file, loads the trained W from the standard grid layout,
recovers per-cell U for N validation images via closed-form NNLS+PGD, ranks concepts
by Sobol' total-order importance, and writes one PNG per class with N rows ×
(top-K + 1) columns (input + top-K concept overlays).

Used to harvest "candidate visualizations" at scale on the cluster: produce one PNG
per class for ~12 multi-object classes, then human-pick the 2-3 best for the rebuttal
figure. No new W training; reuses whatever lives under --grid-root.

Usage (cluster):
  python scripts/visualize_concepts_batch.py \
      --backbone resnet34 \
      --classes-file rebuttal_classes.txt \
      --n-imgs 15 --top-k 5 \
      --output-dir $LAD_OUTPUT_ROOT/visualizations/rebuttal_resnet34
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
import paths  # type: ignore  # noqa: E402
from lad.backbones import load_backbone, make_g_and_h2d
from lad.clip_concepts import load_concepts_txt
from lad.data import (
    ImagenetImageNpzDataset,
    build_paired_items,
    collate_concept_npz,
    npz_suffix_from_variant,
    per_class_kfold,
)
from lad.decomposition import flatten_hw_rowmajor, nonneg_ls_init_U_hybrid
from lad.metrics import estimate_importance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--classes-file", type=Path, required=True,
                   help="One class name per line.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--npz-variant", default="ViT-B-16_r16_14x14")
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-imgs", type=int, default=15,
                   help="How many validation images per class to render.")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--grid-root", type=Path, default=paths.GRID_OUTPUT_ROOT)
    p.add_argument("--mapping-csv", type=Path,
                   default=_REPO / "concept_outputs" / "imagenet1k_classes.csv")
    p.add_argument("--concepts-dir", type=Path, default=paths.CONCEPTS_DIR)
    p.add_argument("--output-dir", type=Path,
                   default=paths.OUTPUT_ROOT / "visualizations" / "batch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dpi", type=int, default=130)
    return p.parse_args()


def visualize_one_class(class_name: str, args, model, g, h_2d, name_to_idx) -> Path | None:
    """Render and save the per-class figure; return the saved path (or None on skip)."""
    device = args.device

    # Locate W
    w_path = (args.grid_root / args.backbone / args.npz_variant
              / f"fold_{args.fold}" / class_name / "W_final.pt")
    if not w_path.exists():
        print(f"  [SKIP] no W at {w_path}")
        return None
    W = torch.load(w_path, map_location=device).to(device).float()  # [C, r]

    # Build paired_items + val split
    images_root = args.filtered_root / args.backbone / "correct"
    npz_root = args.npz_root / args.backbone / f"clip_local_P_{args.npz_variant}" / "correct"
    suffix = npz_suffix_from_variant(args.npz_variant)
    if not (images_root / class_name).exists() or not (npz_root / class_name).exists():
        print(f"  [SKIP] missing images or NPZ tree for {class_name}")
        return None
    _classes, _c2i, paired, _ = build_paired_items(images_root, npz_root, suffix, classes=[class_name])
    if not paired:
        print(f"  [SKIP] no paired items for {class_name}")
        return None
    _train_idx, val_idx = per_class_kfold(paired, args.folds, args.fold,
                                          val_ratio=args.val_ratio, seed=args.seed)
    if not val_idx:
        # Fall back to first N indices if val empty (small datasets)
        val_idx = list(range(min(args.n_imgs, len(paired))))

    chosen = val_idx[: args.n_imgs]
    n = len(chosen)
    if n == 0:
        print(f"  [SKIP] zero images selected for {class_name}")
        return None

    # Load images via the same transform as eval
    ds = ImagenetImageNpzDataset(paired)
    items = [ds[i] for i in chosen]
    batch = collate_concept_npz(items)
    imgs, _ys, _P, *_rest = batch
    imgs = imgs.to(device)

    # Forward + NNLS reconstruction
    with torch.inference_mode():
        Z, side = g(imgs)
    A_flat, _idx, hw = flatten_hw_rowmajor(Z)
    h_z, w_z = hw
    U_flat = nonneg_ls_init_U_hybrid(A_flat, W, pgd_iters=30)
    U_BHWR = U_flat.reshape(n, h_z, w_z, args.rank).cpu().numpy()

    # Sobol' importance on per-image-pooled U → top-K
    target = name_to_idx.get(class_name)
    if target is None:
        print(f"  [SKIP] {class_name} not in mapping CSV")
        return None
    U_per_image = U_flat.reshape(n, h_z * w_z, args.rank).mean(dim=1).cpu().numpy()
    importance = estimate_importance(
        U_per_image, W.t().cpu().numpy(), h_2d, h_2d, target,
        batch_size=64, number_of_concepts=args.rank, device=device,
        cls_per_image=side.cpu() if side is not None else None,
    )
    top_idx = np.argsort(-importance)[: args.top_k]

    # Concept names
    concepts_path = args.concepts_dir / f"{class_name}.txt"
    concepts = (load_concepts_txt(concepts_path) if concepts_path.exists()
                else [f"concept_{k}" for k in range(args.rank)])

    # Reload PIL images at native res for display
    val_img_paths = [paired[i][0] for i in chosen]
    pils = [Image.open(p).convert("RGB").resize((224, 224)) for p in val_img_paths]

    # Render
    n_cols = args.top_k + 1
    fig, axes = plt.subplots(n, n_cols, figsize=(2.0 * n_cols, 2.0 * n))
    if n == 1:
        axes = axes[None, :]

    for r in range(n):
        axes[r, 0].imshow(pils[r])
        axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title("input", fontsize=9)

        for c_idx, k in enumerate(top_idx, 1):
            heat = U_BHWR[r, :, :, k]
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            axes[r, c_idx].imshow(pils[r], alpha=0.55)
            axes[r, c_idx].imshow(
                heat, cmap="jet", alpha=0.55,
                extent=[0, 224, 224, 0], interpolation="bilinear",
            )
            axes[r, c_idx].axis("off")
            if r == 0:
                title = concepts[k][:22] if k < len(concepts) else f"#{k}"
                axes[r, c_idx].set_title(title, fontsize=8)

    fig.suptitle(f"{args.backbone} / {class_name} — top-{args.top_k} concepts (Sobol)",
                 fontsize=11, y=1.0)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{args.backbone}__{class_name}__top{args.top_k}.png"
    fig.savefig(out_file, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def main():
    args = parse_args()
    classes = [ln.strip() for ln in args.classes_file.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
    print(f"Visualizing {len(classes)} classes for backbone={args.backbone}, "
          f"n_imgs={args.n_imgs}, top_k={args.top_k}")

    name_to_idx: dict[str, int] = {}
    with args.mapping_csv.open() as f:
        for row in csv.DictReader(f):
            name_to_idx[row["simplified"]] = int(row["idx"])

    print(f"Loading {args.backbone} on {args.device} ...")
    model = load_backbone(args.backbone, device=args.device)
    g, h_2d = make_g_and_h2d(model, args.backbone, device=args.device)

    n_ok = 0
    for cls in classes:
        print(f"[{cls}] ...")
        try:
            saved = visualize_one_class(cls, args, model, g, h_2d, name_to_idx)
            if saved is not None:
                print(f"  saved: {saved}")
                n_ok += 1
        except Exception as exc:
            print(f"  [FAIL] {exc!r}")

    print(f"\nDone. Wrote {n_ok}/{len(classes)} class visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
