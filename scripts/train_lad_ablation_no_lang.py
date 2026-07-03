"""Module ablation #1: LAD without the language anchor.

Replaces LAD's "fixed CLIP-anchored S, learn only W" with **standard unsupervised NMF**
on the same spatially-unfolded encoder activations. Everything else (input space,
spatial unfolding, evaluation, classifier head) is identical to LAD — so the row's
delta against full LAD isolates the contribution of the language-guidance module
(paper Sec 3.2: "Semantic Coefficient Substitution").

The basis ``W`` is saved in LAD's ``[C, r]`` convention so ``06_eval_metrics.py
--grid-root outputs/grid_ablation_nolang`` evaluates it with identical metrics.

Default output:
    outputs/grid_ablation_nolang/<backbone>/<npz_variant>/fold_<fold>/<class>/W_final.pt
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
from lad.data import (
    ImagenetImageNpzDataset,
    build_paired_items,
    collate_concept_npz,
    npz_suffix_from_variant,
    per_class_kfold,
)
from lad.decomposition import flatten_hw_rowmajor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--class", dest="class_name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npz-variant", default="ViT-B-16_r16_14x14",
                   help="Used only for paired-items lookup (NPZ content unused — no language).")
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--init", default="nndsvd",
                   choices=["nndsvd", "nndsvda", "nndsvdar", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--output-root", type=Path,
                   default=paths.OUTPUT_ROOT / "grid_ablation_nolang")
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
    # Note: NPZ content is *unused* (this is the no-language ablation). We still require
    # the NPZ existence for paired_items so the train/val split matches LAD exactly.
    if not (npz_root / args.class_name).exists():
        raise FileNotFoundError(f"No NPZ files for {args.class_name!r} under {npz_root} "
                                "(needed only to align the train/val split with LAD)")

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

    print("Collecting spatially-unfolded encoder activations A_bar ...")
    A_chunks = []
    with torch.inference_mode():
        for imgs, *_rest in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            Z, _side = g(imgs)  # [B, C, h, w]
            A_flat, _idx, _hw = flatten_hw_rowmajor(Z)  # [B*L, C]
            A_chunks.append(A_flat.cpu())
    A = torch.cat(A_chunks, 0).float()
    print(f"A_bar: {tuple(A.shape)}   rank: {args.rank}   (no language anchor)")

    print("Running unsupervised NMF on A_bar (no S, no language) ...")
    from sklearn.decomposition import NMF
    A_np = A.detach().cpu().clamp_min(0).numpy()  # NMF requires non-negative
    nmf = NMF(n_components=args.rank, init=args.init,
              max_iter=args.max_iter, tol=args.tol, random_state=args.seed)
    t0 = time.perf_counter()
    _U = nmf.fit_transform(A_np)  # [N, r] — discarded; recomputed at eval via NNLS
    elapsed = time.perf_counter() - t0
    H = nmf.components_  # [r, C]
    W = torch.from_numpy(H.T).float()  # [C, r] — LAD convention
    print(f"NMF done in {elapsed:.1f}s  iters={nmf.n_iter_}  recon_err={nmf.reconstruction_err_:.4f}")

    out_dir = args.output_root / args.backbone / args.npz_variant / f"fold_{args.fold}" / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(W.detach().cpu(), out_dir / "W_final.pt")
    history = {
        "method": "lad_ablation_no_language",
        "backbone": args.backbone, "class": args.class_name, "fold": args.fold,
        "npz_variant": args.npz_variant, "rank": args.rank,
        "max_iter": args.max_iter, "tol": args.tol, "init": args.init,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "n_samples": int(A.shape[0]),
        "reconstruction_err_": float(nmf.reconstruction_err_),
        "n_iter_": int(nmf.n_iter_),
        "elapsed_s": elapsed,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Wrote {out_dir / 'W_final.pt'}")


if __name__ == "__main__":
    main()
