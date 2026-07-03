"""Step 6: walk every trained ``W_final.pt`` and emit one CSV row per (backbone, class, fold).

Columns: backbone, npz_variant, class, fold, n_val,
         acc_recon (Acc), c_ins_auc (C-Ins), c_del_aopc, mse, kl,
         gini, sparsity, elapsed_s.

This script reuses :func:`lad.metrics.compute_accuracy_after_nmf`,
:func:`compute_insertion_auc`, :func:`compute_deletion_score`,
:func:`estimate_importance`, and the spatial concept-insertion/deletion variants where
applicable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

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
from lad.decomposition import flatten_hw_rowmajor, kl_logits, nonneg_ls_init_U_hybrid
from lad.metrics import (
    calculate_gini,
    compute_deletion_score,
    compute_insertion_auc,
    compute_sparsity,
    concept_deletion,
    concept_insertion,
    estimate_importance,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid-root", type=Path, default=paths.GRID_OUTPUT_ROOT)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--output", type=Path, default=paths.OUTPUT_ROOT / "all_classes_results.csv")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, evaluate only the first N (backbone,class,fold) tuples.")
    p.add_argument("--include-folds", type=str, default=None,
                   help="Comma-separated fold indices to evaluate (default: all).")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="This shard's index (0-based).")
    p.add_argument("--shard-count", type=int, default=1,
                   help="Total shard count. Targets are sliced [idx::count] after sorting.")
    p.add_argument("--vit-cls-mode", choices=["preserve", "mean"], default="preserve",
                   help="ViT-only: preserve = keep CLS from real patches (interp a); "
                        "mean = overwrite CLS with mean of reconstructed patches (interp b).")
    p.add_argument("--input-size", type=int, default=224,
                   help="Image resolution fed to the encoder (448 for the _448 backbone).")
    p.add_argument("--mapping-csv", type=Path, default=None,
                   help="CSV with idx,raw,simplified columns mapping folder names to the "
                        "global class index. Default: concept_outputs/imagenet1k_classes.csv. "
                        "For Places365 pass concept_outputs/places365_classes.csv.")
    return p.parse_args()


def discover_w_files(grid_root: Path):
    for w in grid_root.rglob("W_final.pt"):
        # Layout: grid/<backbone>/<variant>/fold_<fold>/<class>/W_final.pt
        try:
            class_name = w.parent.name
            fold = int(w.parent.parent.name.split("_")[1])
            variant = w.parent.parent.parent.name
            backbone = w.parent.parent.parent.parent.name
        except Exception:
            continue
        yield backbone, variant, fold, class_name, w


def load_simplified_to_global(path: Path | None = None) -> dict[str, int]:
    """Load the simplified-name → global class-idx map. Default: ImageNet 1000-class."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "concept_outputs" / "imagenet1k_classes.csv"
    out: dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["simplified"]] = int(row["idx"])
    return out


def evaluate_one(
    backbone: str, variant: str, fold: int, class_name: str, W_path: Path,
    *, args, device, simplified_to_global: dict[str, int],
):
    images_root = args.filtered_root / backbone / "correct"
    npz_root = args.npz_root / backbone / f"clip_local_P_{variant}" / "correct"
    if not (images_root / class_name).exists() or not (npz_root / class_name).exists():
        return None
    if class_name not in simplified_to_global:
        raise KeyError(f"class {class_name!r} not in imagenet1k_classes.csv")
    target_label = simplified_to_global[class_name]
    suffix = npz_suffix_from_variant(variant)
    _classes, _c2i, paired, _ = build_paired_items(images_root, npz_root, suffix, classes=[class_name])
    if not paired:
        return None
    _train_idx, val_idx = per_class_kfold(paired, args.folds, fold, val_ratio=args.val_ratio, seed=args.seed)
    if not val_idx:
        return None

    val_loader = DataLoader(
        Subset(ImagenetImageNpzDataset(paired, transform=imagenet_transform(args.input_size)), val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=device.startswith("cuda"), collate_fn=collate_concept_npz,
    )

    model = load_backbone(backbone, device=device)
    g, h_2d = make_g_and_h2d(model, backbone, device=device, vit_cls_mode=args.vit_cls_mode)
    W = torch.load(W_path, map_location=device).to(device).float()  # [C, r]

    A_chunks, U_chunks, side_chunks = [], [], []
    B = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for imgs, _ys, _P, *_rest in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            Z, side = g(imgs)  # Z: [B, C, h_z, w_z]; side = None (CNN) or CLS (ViT)
            A_flat, _img_idx, hw = flatten_hw_rowmajor(Z)
            U_flat = nonneg_ls_init_U_hybrid(A_flat, W, pgd_iters=30)
            A_chunks.append(A_flat.cpu())
            U_chunks.append(U_flat.cpu())
            if side is not None:
                side_chunks.append(side.cpu())
            B += imgs.size(0)

    A = torch.cat(A_chunks, 0).to(device)
    U = torch.cat(U_chunks, 0).to(device)
    side_full = torch.cat(side_chunks, 0).to(device) if side_chunks else None  # [B, 1, C_cls] or None
    L = A.shape[0] // B
    h_z, w_z = hw
    r = W.shape[1]

    # Reconstruction accuracy + MSE + KL
    A_hat = (U @ W.T).reshape(B, h_z, w_z, -1).permute(0, 3, 1, 2)
    A_orig = A.reshape(B, h_z, w_z, -1).permute(0, 3, 1, 2)
    with torch.inference_mode():
        logits_orig = h_2d(A_orig, side_full) if side_full is not None else h_2d(A_orig)
        logits_hat = h_2d(A_hat, side_full) if side_full is not None else h_2d(A_hat)
        mse = F.mse_loss(A_hat, A_orig).item()
        kl = kl_logits(logits_orig, logits_hat).item()
        acc_recon = float((logits_hat.argmax(-1).cpu() == target_label).float().mean().item())

    # Sparsity / Gini on per-image pooled U
    U_per_image = U.reshape(B, L, r).mean(dim=1)
    sparsity = compute_sparsity(U_per_image)

    # Concept importance via Sobol' (CRAFT-style) on the pooled U
    importance = estimate_importance(
        U_per_image.cpu().numpy(), W.T.cpu().numpy(), h_2d, h_2d, target_label,
        batch_size=64, number_of_concepts=r, device=device,
        cls_per_image=side_full,
    )
    gini = calculate_gini(importance)

    # Concept insertion / deletion
    U_spatial = U.reshape(B, h_z, w_z, r).cpu()
    W_for_io = W.T.cpu()  # [r, C] — old function expected the transpose
    ins = concept_insertion(U_spatial, W_for_io, h_2d, target_label, importance, cls=side_full)
    delete = concept_deletion(U_spatial, W_for_io, h_2d, target_label, importance, cls=side_full)
    c_ins_auc = compute_insertion_auc(ins)
    c_del_aopc = compute_deletion_score(delete)

    elapsed = time.perf_counter() - t0
    return {
        "backbone": backbone, "npz_variant": variant, "class_name": class_name, "fold": fold,
        "n_val": B, "acc_recon": acc_recon, "c_ins_auc": c_ins_auc, "c_del_aopc": c_del_aopc,
        "mse": mse, "kl": kl, "gini": gini, "sparsity": sparsity, "elapsed_s": elapsed,
    }


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    targets = list(discover_w_files(args.grid_root))
    targets.sort(key=lambda t: (t[0], t[1], t[2], t[3]))  # deterministic for sharding
    if args.include_folds:
        keep = {int(x) for x in args.include_folds.split(",")}
        targets = [t for t in targets if t[2] in keep]
    if args.limit:
        targets = targets[: args.limit]
    if args.shard_count > 1:
        targets = targets[args.shard_idx :: args.shard_count]
    print(
        f"Shard {args.shard_idx}/{args.shard_count}: "
        f"{len(targets)} W_final.pt files to evaluate -> {args.output}"
    )

    simplified_to_global = load_simplified_to_global(args.mapping_csv)

    n_ok = 0
    writer = None
    with args.output.open("w", newline="") as f:
        for backbone, variant, fold, cls, w_path in tqdm(targets):
            try:
                row = evaluate_one(backbone, variant, fold, cls, w_path,
                                   args=args, device=args.device,
                                   simplified_to_global=simplified_to_global)
                if row is None:
                    continue
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                n_ok += 1
                print(f"[OK] {backbone}/{variant}/fold_{fold}/{cls}")
            except Exception as exc:
                print(f"[FAIL] {backbone}/{variant}/fold_{fold}/{cls}: {exc!r}")
    print(f"Wrote {n_ok} rows to {args.output}")


if __name__ == "__main__":
    main()
