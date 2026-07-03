"""Evaluation metrics for concept-discovery methods.

Single canonical home for everything previously split across ``metrics.py``,
``C_gini.py`` (the spatial variants), ``sobal_inportance.py``, and ``utility.py``.

Public API
----------
* :func:`torch_to_numpy` — universal tensor → numpy converter.
* :func:`compute_accuracy_after_nmf` — reconstruction accuracy after projecting through ``W``.
* :func:`evaluate_nmf_projection` — joint MSE + KL after re-projection.
* :func:`compute_sparsity` — per-element non-zero rate of ``U``.
* :func:`compute_gini_index`, :func:`calculate_gini` — Gini sparsity of importance scores.
* :func:`estimate_importance` — pooled Sobol' importance (per CRAFT, paper Sec. 4.2).
* :func:`estimate_importance_spatial` — spatial Sobol' importance with patch averaging.
* :func:`sobol_concept_importance_spatial` — image-level Jansen estimator.
* :func:`concept_insertion`, :func:`concept_deletion` — image-level (legacy notebook variant).
* :func:`concept_insertion_spatial`, :func:`concept_deletion_spatial` — patch-level (paper variant).
* :func:`compute_insertion_auc`, :func:`compute_deletion_score` — AUC / AOPC summaries.
* :func:`extract_w_list`, :func:`compute_stability` — cross-fold stability via Hungarian matching.
"""

from __future__ import annotations

from math import ceil
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines.craft.estimators import JansenEstimator
from .baselines.craft.sampler import HaltonSequence
from .decomposition import (
    flatten_hw_rowmajor,
    kl_logits,
    nonneg_ls_init_U,
    unflatten_to_image,
)


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------


def torch_to_numpy(tensor) -> np.ndarray:
    """Detach + move to CPU + convert to numpy. Falls back to ``np.array`` for non-tensors."""
    try:
        return tensor.detach().cpu().numpy()
    except AttributeError:
        return np.array(tensor)


def _batch_inference(model_fn, dataset, batch_size: int = 128, resize=None, device: str = "cuda") -> torch.Tensor:
    """Apply ``model_fn`` to ``dataset`` in batches. Accepts numpy or tensor input."""
    if not isinstance(dataset, torch.Tensor):
        dataset = torch.as_tensor(dataset, dtype=torch.float32)
    nb_batches = ceil(len(dataset) / batch_size)
    out = []
    with torch.no_grad():
        for i in range(nb_batches):
            x = dataset[i * batch_size : (i + 1) * batch_size].to(device)
            if resize:
                x = F.interpolate(x, size=resize, mode="bilinear", align_corners=False)
            out.append(model_fn(x).cpu())
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Reconstruction accuracy and joint MSE+KL (paper "Acc" metric)
# ---------------------------------------------------------------------------


def transform_to_nmf_basis(images_preprocessed, W: torch.Tensor, g: nn.Module, device: str) -> torch.Tensor:
    """Project pooled activations onto the concept basis (matrix multiply, no NNLS)."""
    activations = _batch_inference(g, images_preprocessed, batch_size=64, device=device).to(device)
    activations_avg = torch.mean(activations, dim=(2, 3))  # GAP
    return activations_avg @ W.T


def compute_accuracy_after_nmf(
    images_preprocessed,
    W: torch.Tensor,
    g: nn.Module,
    h_2d: nn.Module,
    imagenet_class: int,
    device: str,
) -> float:
    U_new = transform_to_nmf_basis(images_preprocessed, W, g, device)
    activations_now = U_new @ W
    pred = h_2d(activations_now)
    c = torch.argmax(pred, dim=-1)
    return float((torch.sum(c == imagenet_class) / len(c)).item())


def evaluate_nmf_projection(
    images_preprocessed,
    W: torch.Tensor,
    g: nn.Module,
    h_2d: nn.Module,
    device: str,
    A_original: torch.Tensor,
    logits_original: torch.Tensor,
) -> Tuple[float, float]:
    activations = _batch_inference(g, images_preprocessed, batch_size=4, device=device).to(device)
    A_avg = torch.mean(activations, dim=(2, 3))
    U_new = A_avg @ torch.linalg.pinv(W @ W.T) @ W
    A_recon = U_new @ W
    logits_recon = h_2d(A_recon)
    mse = F.mse_loss(A_recon, A_original)
    kl = kl_logits(logits_original, logits_recon)
    return float(mse.item()), float(kl.item())


# ---------------------------------------------------------------------------
# Sparsity / Gini
# ---------------------------------------------------------------------------


def compute_sparsity(U) -> float:
    if not isinstance(U, torch.Tensor):
        U = torch.as_tensor(U)
    U_flat = U.view(-1, U.shape[-1])
    nz = (U_flat != 0).float().sum(dim=1)
    return float((nz / U.shape[-1]).mean().item())


def compute_gini_index(concept_importance: torch.Tensor) -> float:
    """Gini coefficient of a 1-D non-negative importance vector. 1 = sparsest, 0 = uniform."""
    if concept_importance.dim() != 1:
        raise ValueError("concept_importance must be 1-D")
    sorted_, _ = torch.sort(concept_importance)
    n = sorted_.numel()
    mean = torch.mean(sorted_)
    if mean == 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_.device)
    num = torch.sum((2 * idx - n - 1) * sorted_)
    den = n * torch.sum(sorted_)
    return float((num / den).item())


def calculate_gini(scores) -> float:
    """Numpy variant: takes ``|scores|`` first, returns float in [0, 1]."""
    if not isinstance(scores, np.ndarray):
        scores = scores.clone().detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    scores = np.abs(np.asarray(scores).flatten())
    total = scores.sum()
    if total == 0:
        return 0.0
    sorted_ = np.sort(scores)
    n = len(sorted_)
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * sorted_) / (n * total))


# ---------------------------------------------------------------------------
# Sobol' importance — pooled (image-mean) variant
# ---------------------------------------------------------------------------


def _batch_inference_with_side(
    h_2d, a_pert, batch_size: int, device: str, side_per_row: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run ``h_2d(a_chunk, side_chunk)`` over chunks; ``side_per_row`` is broadcast 1-to-1 with ``a_pert``.

    Used for the ViT path in ``estimate_importance`` where each perturbation needs the
    image's CLS state. If ``side_per_row`` is None, falls back to ``h_2d(a_chunk)``.
    """
    if not isinstance(a_pert, torch.Tensor):
        a_pert = torch.as_tensor(a_pert, dtype=torch.float32)
    n = a_pert.shape[0]
    out = []
    with torch.no_grad():
        for i in range(ceil(n / batch_size)):
            sl = slice(i * batch_size, (i + 1) * batch_size)
            a_chunk = a_pert[sl].to(device)
            if side_per_row is None:
                out.append(h_2d(a_chunk).cpu())
            else:
                out.append(h_2d(a_chunk, side_per_row[sl].to(device)).cpu())
    return torch.cat(out, dim=0)


def estimate_importance(
    U: torch.Tensor,
    W: torch.Tensor,
    h: nn.Module,
    h_2d: nn.Module,
    class_id: int,
    batch_size: int,
    number_of_concepts: int,
    device: str,
    nb_design: int = 32,
    cls_per_image: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Per-concept Sobol' total-order importance (Halton + Jansen estimator).

    Shapes
    ------
    Either ``U: [N, r]`` (uses pooled features) or ``U: [N, H, W, r]`` (spatial).

    Parameters
    ----------
    cls_per_image : optional ``[N, 1, C_cls]`` tensor with the per-image CLS / side-state
        for transformer backbones. Required for ``vit_b_16``; ignored otherwise.
    """
    masks = HaltonSequence()(number_of_concepts, nb_design=nb_design).astype(np.float32)
    estimator = JansenEstimator()
    importances = []

    if U.ndim == 2:
        for i, u in enumerate(U):
            u_pert = u[None, :] * masks
            a_pert = u_pert @ W                       # [nb_design, C] — pooled features
            a_pert = a_pert[:, :, None, None]         # → [nb_design, C, 1, 1] so h_2d's GAP no-ops
            if cls_per_image is not None:
                cls_i = cls_per_image[i:i + 1].expand(a_pert.shape[0], -1, -1)
                y_pred = _batch_inference_with_side(h_2d, a_pert, batch_size, device, cls_i)[:, class_id]
            else:
                y_pred = _batch_inference(h_2d, a_pert, batch_size, device=device)[:, class_id]
            stis = estimator(torch_to_numpy(masks), torch_to_numpy(y_pred), nb_design)
            importances.append(stis)
    elif U.ndim == 4:
        for i, u in enumerate(U):
            u_pert = u[None, :] * masks[:, None, None, :]
            a_pert = np.reshape(u_pert, (-1, u.shape[-1])) @ W
            a_pert = np.reshape(a_pert, (len(masks), U.shape[1], U.shape[2], -1))
            a_pert = np.moveaxis(a_pert, -1, 1)
            if cls_per_image is not None:
                cls_i = cls_per_image[i:i + 1].expand(a_pert.shape[0], -1, -1)
                y_pred = _batch_inference_with_side(h_2d, a_pert, batch_size, device, cls_i)[:, class_id]
            else:
                y_pred = _batch_inference(h, a_pert, batch_size, device=device)[:, class_id]
            stis = estimator(torch_to_numpy(masks), torch_to_numpy(y_pred), nb_design)
            importances.append(stis)
    else:
        raise ValueError("U must be 2-D or 4-D")
    return np.mean(importances, 0)


def estimate_importance_spatial(
    U: torch.Tensor,
    W: torch.Tensor,
    h: nn.Module,
    h_2d: nn.Module,
    class_id: int,
    batch_size: int,
    number_of_concepts: int,
    device: str,
    S: int = 49,
    nb_design: int = 32,
) -> np.ndarray:
    """Spatial Sobol' importance: pools logits over ``S`` patches per image before estimation."""
    N_total = U.shape[0]
    masks = HaltonSequence()(number_of_concepts, nb_design=nb_design).astype(np.float32)
    W_np = torch_to_numpy(W)
    all_stis = []

    for i_start in range(0, N_total, S):
        U_image = U[i_start : i_start + S]
        U_pert = np.stack([u[None, :] * masks for u in U_image], axis=0)
        U_pert = np.swapaxes(U_pert, 0, 1)
        A_pert = U_pert @ W_np
        # Reshape assumes 7×7 spatial layout — generalize via H, W parameters if needed.
        A_pert = A_pert.reshape(masks.shape[0], 7, 7, W_np.shape[1])
        A_pert = np.transpose(A_pert, (0, 3, 1, 2))
        Y = _batch_inference(h_2d, A_pert, batch_size, device=device)
        Y_target = Y[:, class_id]
        Y_pooled = Y_target.reshape(masks.shape[0], S).mean(dim=1)
        stis = JansenEstimator()(torch_to_numpy(masks), torch_to_numpy(Y_pooled), nb_design)
        all_stis.append(stis)
    return np.mean(all_stis, 0)


# ---------------------------------------------------------------------------
# Spatial Jansen estimator (image-level, batched, with target re-eval — paper variant)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predict_logits_batched_spatial(
    h_2d: nn.Module,
    A_flat: torch.Tensor,
    img_idx: torch.Tensor,
    B: int,
    hw: Tuple[int, int],
    bs: int = 64,
) -> torch.Tensor:
    outs = []
    for i in range(0, B, bs):
        mask = (img_idx >= i) & (img_idx < (i + bs))
        if not mask.any():
            continue
        A_batch = A_flat[mask]
        idx_batch = img_idx[mask] - i
        B_batch = min(bs, B - i)
        A_spatial = unflatten_to_image(A_batch, idx_batch, B_batch, hw)
        outs.append(h_2d(A_spatial))
    return torch.cat(outs, 0)


@torch.no_grad()
def _f_mean_prob_spatial(
    h_2d: nn.Module,
    U_mat_spatial: torch.Tensor,
    W: torch.Tensor,
    y_rows_img: torch.Tensor,
    hw: Tuple[int, int],
    bs_logits: int = 64,
) -> torch.Tensor:
    k, L, _ = U_mat_spatial.shape
    device = U_mat_spatial.device
    U_flat = U_mat_spatial.reshape(k * L, U_mat_spatial.shape[-1])
    A_hat_flat = U_flat @ W.T
    img_idx = torch.arange(k, device=device).repeat_interleave(L)
    logits = _predict_logits_batched_spatial(h_2d, A_hat_flat, img_idx, k, hw, bs=bs_logits)
    probs = F.softmax(logits, dim=1)
    return probs[torch.arange(k, device=device), y_rows_img].mean()


@torch.no_grad()
def sobol_concept_importance_spatial(
    h_2d: nn.Module,
    U_flat: torch.Tensor,
    W: torch.Tensor,
    A_flat: torch.Tensor,
    y_img: Optional[torch.Tensor],
    B: int,
    L: int,
    hw: Tuple[int, int],
    n_samples: int = 1024,
    batch_logits: int = 64,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Image-level Jansen estimator returning ``(S_first, S_total)`` for r concepts."""
    device = U_flat.device
    N0, r = U_flat.shape
    if N0 != B * L:
        raise ValueError(f"U_flat shape {U_flat.shape} mismatched with B*L={B*L}")
    gen = torch.Generator(device=device).manual_seed(seed)

    U_spatial = U_flat.reshape(B, L, r)
    img_idx_full = torch.arange(B, device=device).repeat_interleave(L)
    if y_img is None:
        logits = _predict_logits_batched_spatial(h_2d, A_flat, img_idx_full, B, hw)
        base_targets = logits.argmax(1)
    else:
        base_targets = y_img.detach().to(device)

    idx_A = torch.randint(0, B, (n_samples,), generator=gen, device=device)
    idx_B = torch.randint(0, B, (n_samples,), generator=gen, device=device)
    U_A = U_spatial[idx_A].clone()
    U_B = U_spatial[idx_B].clone()
    y_A = base_targets[idx_A]
    y_B = base_targets[idx_B]

    K = max(8, min(64, n_samples // 64))
    splits = torch.chunk(torch.arange(n_samples, device=device), K)

    fA_vals = np.array(
        [_f_mean_prob_spatial(h_2d, U_A[sp], W, y_A[sp], hw, bs_logits=batch_logits).item() for sp in splits],
        dtype=np.float64,
    )
    var_fA = float(max(np.var(fA_vals, ddof=1), 1e-12))

    S_first = np.zeros(r, dtype=np.float64)
    S_total = np.zeros(r, dtype=np.float64)
    for i in range(r):
        U_A_Bi = U_A.clone()
        U_A_Bi[:, :, i] = U_B[:, :, i]
        diffs_B_minus_ABi, diffs_A_minus_ABi = [], []
        for sp in splits:
            fB = _f_mean_prob_spatial(h_2d, U_B[sp], W, y_B[sp], hw, bs_logits=batch_logits).item()
            fABi = _f_mean_prob_spatial(h_2d, U_A_Bi[sp], W, y_A[sp], hw, bs_logits=batch_logits).item()
            fA = _f_mean_prob_spatial(h_2d, U_A[sp], W, y_A[sp], hw, bs_logits=batch_logits).item()
            diffs_B_minus_ABi.append((fB - fABi) ** 2)
            diffs_A_minus_ABi.append((fA - fABi) ** 2)
        E_B_ABi = float(np.mean(diffs_B_minus_ABi))
        E_A_ABi = float(np.mean(diffs_A_minus_ABi))
        S_first[i] = max(0.0, 1.0 - E_B_ABi / (2.0 * var_fA))
        S_total[i] = min(1.0, E_A_ABi / (2.0 * var_fA))
    return S_first, S_total


# ---------------------------------------------------------------------------
# Concept insertion / deletion — image-level legacy variant
# ---------------------------------------------------------------------------


def _to_cuda_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to("cuda").detach()


def _call_h2d(h_2d, z, cls):
    """Call ``h_2d(z)`` or ``h_2d(z, cls)`` depending on whether a side state was provided."""
    if cls is None:
        return h_2d(z)
    return h_2d(z, cls)


def concept_deletion(
    images_u, W, h_2d, imagenet_class, concept_importance,
    cls: Optional[torch.Tensor] = None,
) -> dict[int, float]:
    images_u = _to_cuda_tensor(images_u)
    W = _to_cuda_tensor(W)
    if cls is not None:
        cls = cls.to("cuda")
    if isinstance(concept_importance, np.ndarray):
        concept_importance = torch.from_numpy(concept_importance)
    concept_importance = concept_importance.to("cuda").detach()
    if isinstance(imagenet_class, torch.Tensor):
        target = imagenet_class.detach().cpu().to(torch.int64)
    else:
        target = torch.tensor(imagenet_class, dtype=torch.int64)

    activations_org = (images_u @ W).permute(0, 3, 1, 2)
    pred_org = _call_h2d(h_2d, activations_org, cls)
    c_org = torch.argmax(pred_org, dim=-1).cpu()
    accuracy_org = (torch.sum(c_org == target).item()) / len(c_org)

    sorted_idx = torch.argsort(concept_importance, descending=True)
    rank = images_u.shape[-1]
    results = {0: accuracy_org}
    for i in range(1, rank + 1):
        u_mod = images_u.clone().detach()
        u_mod[..., sorted_idx[:i]] = 0
        u_mod = u_mod.contiguous()
        UW = (u_mod @ W).permute(0, 3, 1, 2)
        c = torch.argmax(_call_h2d(h_2d, UW, cls), dim=-1).cpu()
        results[i] = (torch.sum(c == target).item()) / len(c)
    return results


def concept_insertion(
    images_u, W, h_2d, imagenet_class, concept_importance,
    cls: Optional[torch.Tensor] = None,
) -> dict[int, float]:
    images_u = _to_cuda_tensor(images_u)
    W = _to_cuda_tensor(W)
    if cls is not None:
        cls = cls.to("cuda")
    if isinstance(concept_importance, np.ndarray):
        concept_importance = torch.from_numpy(concept_importance)
    concept_importance = concept_importance.to("cuda").detach()
    if isinstance(imagenet_class, torch.Tensor):
        target = imagenet_class.detach().cpu().to(torch.int64)
    else:
        target = torch.tensor(imagenet_class, dtype=torch.int64)

    u_insert = torch.zeros_like(images_u).to("cuda")
    UW_org = (u_insert @ W).permute(0, 3, 1, 2)
    pred_org = _call_h2d(h_2d, UW_org, cls)
    c_org = torch.argmax(pred_org, dim=-1).cpu()
    results = {0: (torch.sum(c_org == target).item()) / len(c_org)}

    sorted_idx = torch.argsort(concept_importance, descending=True)
    rank = images_u.shape[-1]
    for i in range(1, rank + 1):
        u_insert[..., sorted_idx[:i]] = images_u[..., sorted_idx[:i]]
        u_insert = u_insert.contiguous()
        UW = (u_insert @ W).permute(0, 3, 1, 2)
        c = torch.argmax(_call_h2d(h_2d, UW, cls), dim=-1).cpu()
        results[i] = (torch.sum(c == target).item()) / len(c)
    return results


def compute_deletion_score(insertion_deletion_score: dict[int, float]) -> float:
    """AOPC for deletion: mean (initial − current) accuracy."""
    counts = np.array(list(insertion_deletion_score.keys()))
    accs = np.array(list(insertion_deletion_score.values()))
    drops = accs[0] - accs
    rank = len(counts) - 1
    return float(np.sum(drops) / rank) if rank else 0.0


def compute_insertion_auc(insertion_score: dict[int, float]) -> float:
    """AUC for insertion: trapezoidal area normalized by concept-count span."""
    counts = np.array(list(insertion_score.keys()))
    accs = np.array(list(insertion_score.values()))
    if counts[-1] == counts[0]:
        return 0.0
    return float(np.trapezoid(accs, counts) / (counts[-1] - counts[0]))


# ---------------------------------------------------------------------------
# Concept insertion / deletion — spatial (paper variant)
# ---------------------------------------------------------------------------


@torch.no_grad()
def concept_insertion_spatial(
    h_2d: nn.Module,
    U_flat: torch.Tensor,
    W: torch.Tensor,
    y_img: torch.Tensor,
    B: int,
    L: int,
    hw: Tuple[int, int],
    importances: torch.Tensor,
    bs_logits: int = 64,
) -> Tuple[float, np.ndarray]:
    """Image-level Concept Insertion (C-Ins). Returns (AOC, per-step probability scores)."""
    r = W.shape[1]
    U_spatial = U_flat.reshape(B, L, r)
    U_inserted = torch.zeros_like(U_spatial)
    scores = [_f_mean_prob_spatial(h_2d, U_inserted, W, y_img, hw, bs_logits=bs_logits).item()]
    sorted_idx = torch.argsort(importances, descending=True)
    for k in range(r):
        U_inserted[:, :, sorted_idx[k]] = U_spatial[:, :, sorted_idx[k]]
        scores.append(_f_mean_prob_spatial(h_2d, U_inserted, W, y_img, hw, bs_logits=bs_logits).item())
    return float(np.mean(scores)), np.array(scores)


@torch.no_grad()
def concept_deletion_spatial(
    h_2d: nn.Module,
    U_flat: torch.Tensor,
    W: torch.Tensor,
    y_img: torch.Tensor,
    B: int,
    L: int,
    hw: Tuple[int, int],
    importances: torch.Tensor,
    bs_logits: int = 64,
) -> Tuple[float, np.ndarray]:
    """Image-level Concept Deletion (C-Del). Returns (AOC, per-step probability scores)."""
    r = W.shape[1]
    U_spatial = U_flat.reshape(B, L, r)
    U_deleted = U_spatial.clone()
    scores = [_f_mean_prob_spatial(h_2d, U_deleted, W, y_img, hw, bs_logits=bs_logits).item()]
    sorted_idx = torch.argsort(importances, descending=True)
    for k in range(r):
        U_deleted[:, :, sorted_idx[k]] = 0.0
        scores.append(_f_mean_prob_spatial(h_2d, U_deleted, W, y_img, hw, bs_logits=bs_logits).item())
    return float(np.mean(scores)), np.array(scores)


# ---------------------------------------------------------------------------
# Stability across folds (Hungarian-matched cosine similarity)
# ---------------------------------------------------------------------------


def extract_w_list(analyzer, images_np, k_folds: int = 5) -> list[torch.Tensor]:
    """Run an analyzer (e.g. KL_NMF) on each KFold and collect the resulting basis matrices."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    w_list: list[torch.Tensor] = []
    for train_idx, _ in kf.split(images_np):
        subset = images_np[train_idx]
        x = analyzer.preprocess_images(subset)
        patches, activations_avg = analyzer.compute_activations(x)
        _U, W, *_ = analyzer.nmf_kl_pgd(patches, activations_avg, x)
        w_list.append(W)
    return w_list


def compute_stability(w_list: list[torch.Tensor]) -> float:
    """Mean Hungarian-matched cosine similarity between every pair of basis matrices."""
    from scipy.optimize import linear_sum_assignment

    sims = []
    for i, W1 in enumerate(w_list):
        W1n = F.normalize(W1, p=2, dim=1).to("cuda")
        for j in range(i + 1, len(w_list)):
            W2n = F.normalize(w_list[j], p=2, dim=1).to("cuda")
            S = W1n @ W2n.T
            cost = 1 - S.detach().cpu().numpy()
            row, col = linear_sum_assignment(cost)
            matched = S[row, col]
            sims.append(float(matched.mean().item()))
    return float(np.mean(sims)) if sims else 0.0


__all__ = [
    "torch_to_numpy",
    "transform_to_nmf_basis",
    "compute_accuracy_after_nmf",
    "evaluate_nmf_projection",
    "compute_sparsity",
    "compute_gini_index",
    "calculate_gini",
    "estimate_importance",
    "estimate_importance_spatial",
    "sobol_concept_importance_spatial",
    "concept_insertion",
    "concept_deletion",
    "compute_insertion_auc",
    "compute_deletion_score",
    "concept_insertion_spatial",
    "concept_deletion_spatial",
    "extract_w_list",
    "compute_stability",
]
