"""Step 9: paper-faithful evaluation of FACE and CRAFT baselines.

Mirrors the eval semantics of the reference repos:

* ``--mode face``  (matches ``D:/Work/XAP CVPR FINALFINAL/FACE/FACE/metrics.py``):
    - recover U via **image-pooled** closed-form NNLS, *without* non-negativity clamp;
    - reconstruct flat ``A_hat = U @ W^T`` and classify via the model's flat head
      (``model.fc`` for ResNet, ``model.classifier`` for MobileNet/ConvNeXt);
    - C-Ins / C-Del are image-level (zero out concepts in U: ``[B, r]``);

* ``--mode craft`` (matches ``D:/Work/XAP CVPR FINALFINAL/FACE/Baselines/CRAFT/craft_evaluation.py``):
    - recover U **per spatial cell** via closed-form NNLS, *without* clamp;
    - reconstruct spatial ``A_hat: [B, C, h, w]`` and classify via ``h_2d`` (GAP + head);
    - C-Ins / C-Del zero entire concepts across all spatial cells.

Output CSV columns match ``06_eval_metrics``:
    backbone, npz_variant, class_name, fold, n_val,
    acc_recon, c_ins_auc, c_del_aopc, mse, kl, gini, sparsity, elapsed_s
so the three CSVs (LAD/06, FACE/09, CRAFT/09) can be concatenated for direct comparison.
"""

from __future__ import annotations

import argparse
import csv
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
from lad.decomposition import flatten_hw_rowmajor, kl_logits
from lad.metrics import (
    calculate_gini,
    compute_deletion_score,
    compute_insertion_auc,
    compute_sparsity,
    estimate_importance,
)


# ----------------------------------------------------------------------------- helpers


def _flat_head_for(model, backbone: str):
    """Return a callable ``[N, C] -> [N, num_classes]`` matching the reference's ``h_2d``."""
    backbone = backbone.lower()
    if backbone.startswith("resnet"):
        return model.fc
    if backbone == "mobilenet_v2":
        return model.classifier
    if backbone.startswith("convnext"):  # tiny / small / small_ham10000
        cls = model.classifier
        return lambda x: cls(x.unsqueeze(-1).unsqueeze(-1))
    raise NotImplementedError(f"flat head not defined for {backbone!r}")


def load_simplified_to_global(path: Path | None = None) -> dict[str, int]:
    if path is None:
        path = _REPO / "concept_outputs" / "imagenet1k_classes.csv"
    out: dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["simplified"]] = int(row["idx"])
    return out


@torch.no_grad()
def _closed_form_NNLS_no_clamp(A: torch.Tensor, W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Closed-form ``U = A @ W @ (W^T W)^{-1}`` (paper Eq. 4 *without* the ReLU clamp).

    Used to match the reference notebooks, which rely on a single un-clamped projection
    rather than NNLS with PGD refinement.
    """
    G = W.t() @ W
    G_inv = torch.linalg.pinv(G + eps * torch.eye(G.shape[0], device=G.device, dtype=G.dtype))
    return A @ W @ G_inv


# --------------------------------------------------------------------- FACE-mode metrics


@torch.no_grad()
def _face_concept_curves(U: torch.Tensor, W: torch.Tensor, flat_head, target_label: int,
                         importance: np.ndarray) -> tuple[dict[int, float], dict[int, float]]:
    """Image-level concept insertion / deletion in FACE eval semantics.

    U : ``[B, r]`` image-pooled coefficients.
    W : ``[C, r]`` LAD-conv basis (transposed at use site to get ``[r, C]``).
    flat_head : ``[N, C] -> [N, num_classes]``.
    """
    Wt = W.t().contiguous()  # [r, C]
    rank = U.shape[-1]
    sorted_idx = torch.from_numpy(np.argsort(-importance)).long().to(U.device)

    # baseline accuracy with full U
    A_hat = U @ Wt
    base_acc = float((flat_head(A_hat).argmax(-1) == target_label).float().mean().item())

    # deletion: progressively zero top-k concepts
    delete: dict[int, float] = {0: base_acc}
    for i in range(1, rank + 1):
        U_mod = U.clone()
        U_mod[:, sorted_idx[:i]] = 0
        c = (flat_head(U_mod @ Wt).argmax(-1) == target_label).float().mean().item()
        delete[i] = float(c)

    # insertion: progressively reveal top-k concepts starting from zeros
    U_ins = torch.zeros_like(U)
    base_ins = float((flat_head(U_ins @ Wt).argmax(-1) == target_label).float().mean().item())
    insert: dict[int, float] = {0: base_ins}
    for i in range(1, rank + 1):
        U_ins[:, sorted_idx[:i]] = U[:, sorted_idx[:i]]
        c = (flat_head(U_ins @ Wt).argmax(-1) == target_label).float().mean().item()
        insert[i] = float(c)

    return insert, delete


def _evaluate_face(W_path, backbone, variant, fold, class_name, *, args, device, target_label):
    images_root = args.filtered_root / backbone / "correct"
    npz_root = args.npz_root / backbone / f"clip_local_P_{variant}" / "correct"
    if not (images_root / class_name).exists() or not (npz_root / class_name).exists():
        return None
    suffix = npz_suffix_from_variant(variant)
    _classes, _c2i, paired, _ = build_paired_items(images_root, npz_root, suffix, classes=[class_name])
    if not paired:
        return None
    _train_idx, val_idx = per_class_kfold(paired, args.folds, fold,
                                          val_ratio=args.val_ratio, seed=args.seed)
    if not val_idx:
        return None

    val_loader = DataLoader(
        Subset(ImagenetImageNpzDataset(paired), val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=device.startswith("cuda"), collate_fn=collate_concept_npz,
    )

    model = load_backbone(backbone, device=device)
    g, h_2d = make_g_and_h2d(model, backbone, device=device)
    flat_head = _flat_head_for(model, backbone)
    W = torch.load(W_path, map_location=device).to(device).float()  # [C, r]

    A_avg_chunks = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for imgs, *_rest in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            Z, _side = g(imgs)
            A_avg_chunks.append(Z.mean(dim=(2, 3)).cpu())
    A_avg = torch.cat(A_avg_chunks, 0).to(device).float()  # [B, C]
    B = A_avg.shape[0]
    r = W.shape[1]

    U = _closed_form_NNLS_no_clamp(A_avg, W)  # [B, r] — un-clamped, may have negatives
    A_hat = U @ W.t()
    with torch.inference_mode():
        logits_orig = flat_head(A_avg)
        logits_hat = flat_head(A_hat)
        mse = F.mse_loss(A_hat, A_avg).item()
        kl = kl_logits(logits_orig, logits_hat).item()
        acc_recon = float((logits_hat.argmax(-1) == target_label).float().mean().item())

    # Sparsity / Gini
    sparsity = compute_sparsity(U)

    # Sobol' importance: estimate_importance reshapes [N, C] → [N, C, 1, 1] internally, so
    # we pass the standard 4-D-input h_2d (its GAP is a no-op on a 1×1 spatial map).
    importance = estimate_importance(
        U.cpu().numpy(), W.t().cpu().numpy(), h_2d, h_2d, target_label,
        batch_size=64, number_of_concepts=r, device=device,
    )
    gini = calculate_gini(importance)

    insert, delete = _face_concept_curves(U, W, flat_head, target_label, importance)
    c_ins_auc = compute_insertion_auc(insert)
    c_del_aopc = compute_deletion_score(delete)

    return {
        "backbone": backbone, "npz_variant": variant, "class_name": class_name, "fold": fold,
        "n_val": B, "acc_recon": acc_recon, "c_ins_auc": c_ins_auc, "c_del_aopc": c_del_aopc,
        "mse": mse, "kl": kl, "gini": gini, "sparsity": sparsity,
        "elapsed_s": time.perf_counter() - t0,
    }


# -------------------------------------------------------------------- CRAFT-mode metrics


@torch.no_grad()
def _craft_concept_curves(U_spatial_BHWR: torch.Tensor, W: torch.Tensor, h_2d, target_label: int,
                          importance: np.ndarray) -> tuple[dict[int, float], dict[int, float]]:
    """Spatial concept insertion / deletion in CRAFT eval semantics.

    U_spatial_BHWR : ``[B, h, w, r]``.
    W : ``[C, r]`` LAD-conv basis.
    h_2d : ``[B, C, h, w] -> [B, num_classes]`` (GAP + head).
    """
    rank = U_spatial_BHWR.shape[-1]
    sorted_idx = torch.from_numpy(np.argsort(-importance)).long().to(U_spatial_BHWR.device)
    Wt = W.t().contiguous()  # [r, C]

    def _classify(U_in):
        # U_in: [B, h, w, r]; W^T: [r, C] → [B, h, w, C] → [B, C, h, w]
        A_hat = (U_in @ Wt).permute(0, 3, 1, 2).contiguous()
        return h_2d(A_hat).argmax(-1)

    base_acc = float((_classify(U_spatial_BHWR) == target_label).float().mean().item())
    delete: dict[int, float] = {0: base_acc}
    for i in range(1, rank + 1):
        U_mod = U_spatial_BHWR.clone()
        U_mod[..., sorted_idx[:i]] = 0
        delete[i] = float((_classify(U_mod) == target_label).float().mean().item())

    U_ins = torch.zeros_like(U_spatial_BHWR)
    insert: dict[int, float] = {0: float((_classify(U_ins) == target_label).float().mean().item())}
    for i in range(1, rank + 1):
        U_ins[..., sorted_idx[:i]] = U_spatial_BHWR[..., sorted_idx[:i]]
        insert[i] = float((_classify(U_ins) == target_label).float().mean().item())
    return insert, delete


def _evaluate_craft(W_path, backbone, variant, fold, class_name, *, args, device, target_label):
    images_root = args.filtered_root / backbone / "correct"
    npz_root = args.npz_root / backbone / f"clip_local_P_{variant}" / "correct"
    if not (images_root / class_name).exists() or not (npz_root / class_name).exists():
        return None
    suffix = npz_suffix_from_variant(variant)
    _classes, _c2i, paired, _ = build_paired_items(images_root, npz_root, suffix, classes=[class_name])
    if not paired:
        return None
    _train_idx, val_idx = per_class_kfold(paired, args.folds, fold,
                                          val_ratio=args.val_ratio, seed=args.seed)
    if not val_idx:
        return None

    val_loader = DataLoader(
        Subset(ImagenetImageNpzDataset(paired), val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=device.startswith("cuda"), collate_fn=collate_concept_npz,
    )

    model = load_backbone(backbone, device=device)
    g, h_2d = make_g_and_h2d(model, backbone, device=device)
    W = torch.load(W_path, map_location=device).to(device).float()  # [C, r]

    A_chunks = []
    hw = None
    t0 = time.perf_counter()
    with torch.inference_mode():
        for imgs, *_rest in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            Z, _side = g(imgs)
            if hw is None:
                hw = (Z.shape[2], Z.shape[3])
            A_flat, _idx, _hw = flatten_hw_rowmajor(Z)
            A_chunks.append(A_flat.cpu())
    A_flat = torch.cat(A_chunks, 0).to(device).float()  # [B*L, C]
    h_z, w_z = hw
    L = h_z * w_z
    B = A_flat.shape[0] // L
    r = W.shape[1]

    # Per-spatial-cell U via no-clamp closed-form NNLS
    U_flat = _closed_form_NNLS_no_clamp(A_flat, W)  # [B*L, r]
    U_BHWR = U_flat.reshape(B, h_z, w_z, r)

    # Reconstruction + image-level acc/MSE/KL on full classifier (h_2d)
    A_hat_4d = (U_flat @ W.t()).reshape(B, h_z, w_z, -1).permute(0, 3, 1, 2).contiguous()
    A_orig_4d = A_flat.reshape(B, h_z, w_z, -1).permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        logits_orig = h_2d(A_orig_4d)
        logits_hat = h_2d(A_hat_4d)
        mse = F.mse_loss(A_hat_4d, A_orig_4d).item()
        kl = kl_logits(logits_orig, logits_hat).item()
        acc_recon = float((logits_hat.argmax(-1) == target_label).float().mean().item())

    # Sparsity on per-image-pooled U (matches reference: mean over h, w)
    U_per_image = U_BHWR.mean(dim=(1, 2))  # [B, r]
    sparsity = compute_sparsity(U_per_image)

    importance = estimate_importance(
        U_per_image.cpu().numpy(), W.t().cpu().numpy(), h_2d, h_2d, target_label,
        batch_size=64, number_of_concepts=r, device=device,
    )
    gini = calculate_gini(importance)

    insert, delete = _craft_concept_curves(U_BHWR, W, h_2d, target_label, importance)
    c_ins_auc = compute_insertion_auc(insert)
    c_del_aopc = compute_deletion_score(delete)

    return {
        "backbone": backbone, "npz_variant": variant, "class_name": class_name, "fold": fold,
        "n_val": B, "acc_recon": acc_recon, "c_ins_auc": c_ins_auc, "c_del_aopc": c_del_aopc,
        "mse": mse, "kl": kl, "gini": gini, "sparsity": sparsity,
        "elapsed_s": time.perf_counter() - t0,
    }


# ------------------------------------------------------------------------------ main


def discover_w_files(grid_root: Path):
    for w in grid_root.rglob("W_final.pt"):
        try:
            class_name = w.parent.name
            fold = int(w.parent.parent.name.split("_")[1])
            variant = w.parent.parent.parent.name
            backbone = w.parent.parent.parent.parent.name
        except Exception:
            continue
        yield backbone, variant, fold, class_name, w


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["face", "craft"],
                   help="face = image-pooled+flat-fc; craft = spatial+full-classifier")
    p.add_argument("--grid-root", type=Path, required=True,
                   help="e.g. outputs/grid_face or outputs/grid_craft")
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include-folds", type=str, default=None)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--mapping-csv", type=Path, default=None,
                   help="idx,raw,simplified CSV mapping folder names to head indices. "
                        "Default: concept_outputs/imagenet1k_classes.csv. For HAM10000 pass "
                        "concept_outputs/ham10000_classes.csv.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    targets = list(discover_w_files(args.grid_root))
    targets.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    if args.include_folds:
        keep = {int(x) for x in args.include_folds.split(",")}
        targets = [t for t in targets if t[2] in keep]
    if args.limit:
        targets = targets[: args.limit]
    if args.shard_count > 1:
        targets = targets[args.shard_idx :: args.shard_count]
    print(f"[mode={args.mode}] Shard {args.shard_idx}/{args.shard_count}: "
          f"{len(targets)} W_final.pt files -> {args.output}")

    simplified_to_global = load_simplified_to_global(args.mapping_csv)
    eval_fn = _evaluate_face if args.mode == "face" else _evaluate_craft

    n_ok = 0
    writer = None
    with args.output.open("w", newline="") as f:
        for backbone, variant, fold, cls, w_path in tqdm(targets):
            if cls not in simplified_to_global:
                print(f"[SKIP] {cls!r} not in imagenet1k_classes.csv")
                continue
            target_label = simplified_to_global[cls]
            try:
                row = eval_fn(w_path, backbone, variant, fold, cls,
                              args=args, device=args.device, target_label=target_label)
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
