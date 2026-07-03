"""Step 5: train the concept basis ``W`` for one (backbone, class, fold) tuple.

This is LAD's main optimization (paper Eq. 1):

    min_{W ≥ 0}  ½ ‖Ā − S W^T‖²_F

where ``S`` is the fixed CLIP-derived per-spatial-position similarity matrix
(provided by Step 4 as per-image NPZs) and ``Ā`` is the unfolded encoder activation map.

The script saves
    outputs/grid/<backbone>/<npz_variant>/fold_<fold>/<class>/W_final.pt
plus a one-row CSV ``train_metrics.csv`` with reconstruction loss, sparsity, and timing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))  # prefer worktree-local lad over any pip-editable install
import paths  # type: ignore  # noqa: E402
from lad.backbones import load_backbone, make_g_and_h2d
from lad.data import (
    ImagenetImageNpzDataset,
    build_paired_items,
    collate_concept_npz,
    imagenet_transform,
    npz_suffix_from_variant,
    per_class_kfold,
)
from lad.decomposition import (
    flatten_hw_rowmajor,
    nndsvd_init,
    train_W_pgd,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--class", dest="class_name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npz-variant", default="ViT-B-16_r16_14x14",
                   help="One of ViT-B-16_r16_14x14, ViT-B-16_r32_7x7.")
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=None,
                   help="If unset, derived from spectral norm of S^T S (paper-faithful).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--output-root", type=Path, default=paths.GRID_OUTPUT_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--input-size", type=int, default=224,
                   help="Image resolution fed to the encoder (448 -> 14x14 grid for the _448 backbone).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    images_root = args.filtered_root / args.backbone / "correct"
    if not (images_root / args.class_name).exists():
        raise FileNotFoundError(f"No filtered images for class {args.class_name!r} under {images_root}")

    npz_root = args.npz_root / args.backbone / f"clip_local_P_{args.npz_variant}" / "correct"
    npz_suffix = npz_suffix_from_variant(args.npz_variant)
    if not (npz_root / args.class_name).exists():
        raise FileNotFoundError(f"No NPZ files for class {args.class_name!r} under {npz_root}")

    classes, class_to_idx, paired, report = build_paired_items(
        images_root, npz_root, npz_suffix, classes=[args.class_name],
    )
    print("paired-items report:", report)
    if not paired:
        raise RuntimeError(f"No paired items for class {args.class_name!r}")
    train_idx, val_idx = per_class_kfold(paired, args.folds, args.fold, val_ratio=args.val_ratio, seed=args.seed)

    full_ds = ImagenetImageNpzDataset(paired, transform=imagenet_transform(args.input_size))
    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"), drop_last=False, collate_fn=collate_concept_npz,
    )

    print(f"Loading backbone {args.backbone} on {device} ...")
    model = load_backbone(args.backbone, device=device)
    g, h_2d = make_g_and_h2d(model, args.backbone, device=device)

    # ---------------- collect (A, S) over the full training subset ----------------
    A_chunks: List[torch.Tensor] = []
    S_chunks: List[torch.Tensor] = []

    print("Collecting A (encoder activations) and S (CLIP similarities) ...")
    with torch.inference_mode():
        for imgs, _y, P, *_rest in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            Z, _side = g(imgs)  # [B, C, h_z, w_z]; side = None for CNNs, CLS for ViT (unused in training)
            A_flat, _img_idx, hw_z = flatten_hw_rowmajor(Z)
            B = imgs.size(0)
            r = P.shape[1]
            # Convert P [B, r, h_p, w_p] → spatial S [B*h_p*w_p, r]; if h_p≠h_z, downsample.
            P = P.float().to(device)
            if P.shape[-2:] != hw_z:
                P = F.adaptive_avg_pool2d(P, hw_z)
            S_flat = P.permute(0, 2, 3, 1).reshape(B * hw_z[0] * hw_z[1], r)
            A_chunks.append(A_flat.cpu())
            S_chunks.append(S_flat.cpu())

    A = torch.cat(A_chunks, 0)
    S = torch.cat(S_chunks, 0)
    print(f"A: {tuple(A.shape)}   S: {tuple(S.shape)}   rank: {args.rank}")
    if S.shape[1] != args.rank:
        raise ValueError(f"NPZ rank ({S.shape[1]}) does not match --rank ({args.rank})")

    # ---------------- train W ----------------
    print("Training W via PGD on Eq. 1 ...")
    A_dev = A.to(device).float()
    S_dev = S.to(device).float()

    # Seed W from NNDSVD on A.T to give a sensible non-negative starting point.
    # NNDSVD(A^T) returns the first factor with shape [C, r] — exactly the basis we want.
    W_seed, _ = nndsvd_init(A_dev.T.contiguous(), args.rank)
    W_init = W_seed.contiguous().to(device)

    t0 = time.perf_counter()
    W, losses = train_W_pgd(A_dev, S_dev, rank=args.rank, n_iter=args.epochs, lr=args.lr, W_init=W_init)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.1f}s. Final loss: {losses[-1]:.6f}")

    # ---------------- save ----------------
    out_dir = args.output_root / args.backbone / args.npz_variant / f"fold_{args.fold}" / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(W.detach().cpu(), out_dir / "W_final.pt")

    history = {
        "backbone": args.backbone,
        "class": args.class_name,
        "fold": args.fold,
        "npz_variant": args.npz_variant,
        "rank": args.rank,
        "epochs": args.epochs,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "elapsed_s": elapsed,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Wrote {out_dir / 'W_final.pt'}")


if __name__ == "__main__":
    main()
