"""Step 8: paper-faithful CRAFT training per (backbone, class, fold).

Replicates the reference's ``Craft.fit`` (Fel et al., CVPR 2023) without external deps:

* unfold each 224×224 image into ``patch_size`` patches with 80% overlap,
* encode each patch through the backbone, mean-pool → ``A: [N_patches, C]``,
* sklearn ``NMF`` → ``U: [N_patches, r]`` (discarded), ``H: [r, C]``.

We save the basis as ``W = H.T : [C, r]`` (LAD convention) so any of the eval scripts
(``06_eval_metrics`` or ``09_eval_baselines``) can pick it up.

Default output:
    outputs/grid_craft/<backbone>/<npz_variant>/fold_<fold>/<class>/W_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
import paths  # type: ignore  # noqa: E402
from lad.backbones import load_backbone, make_g_and_h2d
from lad.baselines.craft_train import train_craft_W
from lad.data import (
    ImagenetImageNpzDataset,
    build_paired_items,
    collate_concept_npz,
    npz_suffix_from_variant,
    per_class_kfold,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--class", dest="class_name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npz-variant", default="ViT-B-16_r16_14x14")
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--stride-frac", type=float, default=0.8)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--init", default="nndsvd",
                   choices=["nndsvd", "nndsvda", "nndsvdar", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--output-root", type=Path, default=paths.OUTPUT_ROOT / "grid_craft")
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
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"), drop_last=False, collate_fn=collate_concept_npz,
    )

    print(f"Loading backbone {args.backbone} on {device} ...")
    model = load_backbone(args.backbone, device=device)
    g, _h_2d = make_g_and_h2d(model, args.backbone, device=device)

    print("Collecting training images (224x224) ...")
    img_chunks = []
    for imgs, *_rest in train_loader:
        img_chunks.append(imgs)
    images_4d = torch.cat(img_chunks, 0).to(device)
    print(f"images_4d: {tuple(images_4d.shape)}   patch_size={args.patch_size}   "
          f"stride={int(args.patch_size * args.stride_frac)}")

    print("Training paper-faithful CRAFT (NMF on patch features) ...")
    t0 = time.perf_counter()
    W, info = train_craft_W(
        images_4d=images_4d,
        g=g,
        rank=args.rank,
        patch_size=args.patch_size,
        stride_frac=args.stride_frac,
        encode_batch_size=args.encode_batch_size,
        n_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.1f}s. n_patches={info['n_patches']}  "
          f"NMF iters={info['n_iter_']}  recon_err={info['reconstruction_err_']:.4f}")

    out_dir = args.output_root / args.backbone / args.npz_variant / f"fold_{args.fold}" / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(W.detach().cpu(), out_dir / "W_final.pt")
    history = {
        "method": "craft_paperfaithful",
        "backbone": args.backbone, "class": args.class_name, "fold": args.fold,
        "npz_variant": args.npz_variant, "rank": args.rank,
        "patch_size": args.patch_size, "stride_frac": args.stride_frac,
        "max_iter": args.max_iter, "tol": args.tol, "init": args.init,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "elapsed_s": elapsed, **info,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Wrote {out_dir / 'W_final.pt'}")


if __name__ == "__main__":
    main()
