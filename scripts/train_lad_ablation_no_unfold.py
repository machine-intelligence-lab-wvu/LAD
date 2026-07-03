"""Module ablation #2: LAD without spatial unfolding.

Replaces LAD's per-spatial-cell ``Ā ∈ ℝ^(nhw × p)`` with image-pooled
``A ∈ ℝ^(n × p)`` (and pools S the same way: per-image mean over the spatial grid).
Everything else (fixed CLIP-anchored S, learn only W via PGD with NNDSVD init,
evaluation, classifier head) is identical to LAD — so the row's delta against full
LAD isolates the contribution of the spatial-unfolding module (paper Sec 3.2:
"Unlike conventional concept discovery methods that operate on globally pooled
representations A ∈ ℝ^(n×p), we perform factorization directly in the spatial
feature space ...").

The basis ``W`` is saved in LAD's ``[C, r]`` convention so ``06_eval_metrics.py
--grid-root outputs/grid_ablation_pooled`` evaluates it with identical metrics
(eval is still spatial — only training is pooled).

Default output:
    outputs/grid_ablation_pooled/<backbone>/<npz_variant>/fold_<fold>/<class>/W_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
import paths  # type: ignore  # noqa: E402
from lad.backbones import load_backbone, make_g_and_h2d
from lad.data import (
    ImagenetImageNpzDataset,
    build_paired_items,
    collate_concept_npz,
    npz_suffix_from_variant,
    per_class_kfold,
)
from lad.decomposition import nndsvd_init, train_W_pgd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--class", dest="class_name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npz-variant", default="ViT-B-16_r16_14x14")
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--output-root", type=Path,
                   default=paths.OUTPUT_ROOT / "grid_ablation_pooled")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    images_root = args.filtered_root / args.backbone / "correct"
    npz_root = args.npz_root / args.backbone / f"clip_local_P_{args.npz_variant}" / "correct"
    if not (images_root / args.class_name).exists():
        raise FileNotFoundError(f"No filtered images for {args.class_name!r} under {images_root}")
    if not (npz_root / args.class_name).exists():
        raise FileNotFoundError(f"No NPZ files for {args.class_name!r} under {npz_root}")

    suffix = npz_suffix_from_variant(args.npz_variant)
    _classes, _c2i, paired, report = build_paired_items(
        images_root, npz_root, suffix, classes=[args.class_name],
    )
    print("paired-items report:", report)
    if not paired:
        raise RuntimeError(f"No paired items for class {args.class_name!r}")
    train_idx, val_idx = per_class_kfold(paired, args.folds, args.fold,
                                         val_ratio=args.val_ratio, seed=args.seed)

    full_ds = ImagenetImageNpzDataset(paired)
    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"), drop_last=False, collate_fn=collate_concept_npz,
    )

    print(f"Loading backbone {args.backbone} on {device} ...")
    model = load_backbone(args.backbone, device=device)
    g, _h_2d = make_g_and_h2d(model, args.backbone, device=device)

    # Collect IMAGE-POOLED activations + IMAGE-POOLED CLIP S (one row per image).
    # This is the "no spatial unfold" ablation: A ∈ ℝ^(n×p) instead of Ā ∈ ℝ^(nhw×p).
    A_chunks, S_chunks = [], []
    print("Collecting image-pooled A and S (no spatial unfold) ...")
    with torch.inference_mode():
        for imgs, _y, P, *_rest in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            Z, _side = g(imgs)  # [B, C, h_z, w_z]
            A_pooled = Z.mean(dim=(2, 3))  # [B, C]
            B = imgs.size(0)
            r = P.shape[1]
            P_dev = P.float().to(device)              # [B, r, h_p, w_p]
            S_pooled = P_dev.mean(dim=(2, 3))         # [B, r]
            A_chunks.append(A_pooled.cpu())
            S_chunks.append(S_pooled.cpu())

    A = torch.cat(A_chunks, 0).to(device).float()  # [N, C]
    S = torch.cat(S_chunks, 0).to(device).float()  # [N, r]
    print(f"A: {tuple(A.shape)}   S: {tuple(S.shape)}   rank: {args.rank}")
    if S.shape[1] != args.rank:
        raise ValueError(f"NPZ rank ({S.shape[1]}) does not match --rank ({args.rank})")

    # Train W with same PGD as LAD, just on much smaller pooled matrices.
    # NNDSVD requires the SVD rank to be >= requested rank; with pooled A having only
    # N=n samples and rank=25, classes with <25 train images can't be NNDSVD-init'd.
    # Fall back to random non-negative init in that case (rare in practice — fold-0 of
    # K=3 typically has hundreds of train images per class).
    print("Training W via PGD on pooled (A, S) ...")
    if A.shape[0] >= args.rank and A.shape[1] >= args.rank:
        W_seed, _ = nndsvd_init(A.T.contiguous(), args.rank)
        W_init = W_seed.contiguous().to(device)
    else:
        print(f"  NNDSVD skipped (A is {tuple(A.shape)}, need both dims >= rank={args.rank}); "
              f"using random non-negative init.")
        W_init = torch.rand(A.shape[1], args.rank, device=device).abs() * 0.1

    t0 = time.perf_counter()
    W, losses = train_W_pgd(A, S, rank=args.rank, n_iter=args.epochs, lr=args.lr, W_init=W_init)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.1f}s. Final loss: {losses[-1]:.6f}")

    out_dir = args.output_root / args.backbone / args.npz_variant / f"fold_{args.fold}" / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(W.detach().cpu(), out_dir / "W_final.pt")
    history = {
        "method": "lad_ablation_no_unfold",
        "backbone": args.backbone, "class": args.class_name, "fold": args.fold,
        "npz_variant": args.npz_variant, "rank": args.rank, "epochs": args.epochs,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "n_samples": int(A.shape[0]),
        "elapsed_s": elapsed,
        "loss_first": losses[0], "loss_last": losses[-1],
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Wrote {out_dir / 'W_final.pt'}")


if __name__ == "__main__":
    main()
