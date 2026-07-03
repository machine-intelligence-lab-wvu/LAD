"""Decomposition kernel: spatial ops, NMF inits, LAD trainer, inference, and the FACE baseline.

This module collapses the previously duplicated content of ``my_utils.py``,
``C_gini.py``, and ``nmfkl.py`` into a single canonical location.

Conventions
-----------
* Encoder activations ``Z`` arrive as ``[B, C, H, W]``.
* Flattened activations ``A`` are ``[B*L, C]`` where ``L = H*W`` (row-major).
* The concept basis ``W`` has shape ``[C, r]`` (channels × concepts).
* The semantic / latent coefficient matrix ``S`` (a.k.a. ``U``) has shape ``[N, r]``.
* Reconstruction is ``Â = S @ W^T``, matching Eq. 1 of the paper after transposing.

Paper equations (LAD)
----------------------
* Eq. 1 (training, fixed S, learn W ≥ 0): ``min_W ½ ‖Ā − S W^T‖²_F``.
* Eq. 4 (inference, closed-form init for Ŝ ≥ 0): ``Ŝ_0 = ReLU(Ā W (W^T W)^{-1})``.
* Eq. 5 (PGD refinement of Ŝ): ``Ŝ ← ReLU(Ŝ − η (Ŝ W^T W − Ā W))``.
"""

from __future__ import annotations

from math import ceil
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial ops (flatten / unflatten / pool / KL on logits)
# ---------------------------------------------------------------------------


def flatten_hw_rowmajor(z4d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Flatten ``[B, C, H, W]`` to ``[B*L, C]`` row-major over (H, W).

    Returns
    -------
    A_flat : ``[B*L, C]`` float tensor on the same device as ``z4d``.
    img_idx : ``[B*L]`` long tensor mapping each row to its image index.
    hw : (H, W).
    """
    B, C, H, W = z4d.shape
    A_flat = z4d.permute(0, 2, 3, 1).reshape(B * H * W, C)
    img_idx = torch.arange(B, device=z4d.device).repeat_interleave(H * W)
    return A_flat, img_idx, (H, W)


@torch.no_grad()
def unflatten_to_image(
    A_flat: torch.Tensor,
    img_idx: torch.Tensor,
    B: int,
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Inverse of :func:`flatten_hw_rowmajor` for the canonical row-major ordering.

    The implementation assumes ``img_idx`` was produced by :func:`flatten_hw_rowmajor`
    so that rows for image ``b`` occupy positions ``b*L : (b+1)*L`` consecutively.
    """
    H, W = hw
    L = H * W
    N, C = A_flat.shape
    if N != img_idx.numel():
        raise ValueError("len(img_idx) must match A_flat rows")
    if N != B * L:
        raise ValueError(f"N={N} does not equal B*L={B*L}")

    img_idx = img_idx.to(dtype=torch.long, device=A_flat.device)
    A_img = torch.zeros(B, H, W, C, device=A_flat.device, dtype=A_flat.dtype)
    pos = torch.arange(L, device=A_flat.device).repeat(B)
    h = torch.div(pos, W, rounding_mode="floor")
    w = pos.remainder(W)
    A_img[img_idx, h, w] = A_flat
    return A_img.permute(0, 3, 1, 2).contiguous()


def pool_by_image(A: torch.Tensor, img_idx: torch.Tensor, B: int) -> torch.Tensor:
    """Mean-pool ``A: [B*L, C]`` to ``[B, C]`` over the spatial dimension via ``img_idx``."""
    C = A.shape[1]
    A_sum = torch.zeros(B, C, device=A.device, dtype=A.dtype)
    cnt = torch.zeros(B, 1, device=A.device, dtype=A.dtype)
    A_sum.index_add_(0, img_idx, A)
    cnt.index_add_(0, img_idx, torch.ones_like(img_idx, dtype=A.dtype).unsqueeze(1))
    return A_sum / cnt.clamp_min(1.0)


def kl_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(softmax(p) ‖ softmax(q)) — convention: ``p`` is the teacher, ``q`` the student."""
    return F.kl_div(F.log_softmax(q_logits, dim=-1), F.softmax(p_logits, dim=-1), reduction="batchmean")


# ---------------------------------------------------------------------------
# Initialization for the basis W and the coefficient matrix S/U
# ---------------------------------------------------------------------------


def nndsvd_init(A: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """NNDSVD initialization (Boutsidis & Gallopoulos, 2008) for non-negative ``A ≈ U W``.

    Parameters
    ----------
    A : ``[N, C]``
    r : rank (number of concepts)

    Returns
    -------
    (U0, W0) both non-negative; shapes ``[N, r]`` and ``[r, C]`` respectively.
    """
    with torch.no_grad():
        U_svd, S_svd, V_svd = torch.linalg.svd(A, full_matrices=False)
        U0 = torch.zeros_like(U_svd[:, :r], dtype=torch.float32)
        W0 = torch.zeros_like(V_svd[:r, :], dtype=torch.float32)

        U0[:, 0] = torch.sqrt(S_svd[0]) * torch.abs(U_svd[:, 0])
        W0[0, :] = torch.sqrt(S_svd[0]) * torch.abs(V_svd[0, :])

        for i in range(1, r):
            u, v = U_svd[:, i], V_svd[i, :]
            u_pos, u_neg = torch.clamp(u, min=0), torch.clamp(-u, min=0)
            v_pos, v_neg = torch.clamp(v, min=0), torch.clamp(-v, min=0)
            u_norm, v_norm = torch.norm(u_pos), torch.norm(v_pos)
            if u_norm * v_norm > torch.norm(u_neg) * torch.norm(v_neg):
                U0[:, i] = torch.sqrt(S_svd[i]) * u_pos / (u_norm + 1e-8)
                W0[i, :] = torch.sqrt(S_svd[i]) * v_pos / (v_norm + 1e-8)
            else:
                U0[:, i] = torch.sqrt(S_svd[i]) * u_neg / (torch.norm(u_neg) + 1e-8)
                W0[i, :] = torch.sqrt(S_svd[i]) * v_neg / (torch.norm(v_neg) + 1e-8)
        U0 = torch.clamp(U0, min=0)
        W0 = torch.clamp(W0, min=0)
    return U0, W0


@torch.no_grad()
def nonneg_ls_init_U(A: torch.Tensor, W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Closed-form NNLS initialization (paper Eq. 4): ``U₀ = ReLU(A W (W^T W)⁻¹)``.

    Parameters
    ----------
    A : ``[N, C]``
    W : ``[C, r]``  (basis with channels × concepts)

    Returns
    -------
    U : ``[N, r]`` non-negative.
    """
    A = A.float()
    W = W.float()
    G = W.t() @ W  # [r, r]
    G_inv = torch.linalg.pinv(G + eps * torch.eye(G.shape[0], device=G.device, dtype=G.dtype))
    U0 = A @ W @ G_inv
    return U0.clamp_min(0)


# Keep an alias matching the paper's notation.
nonneg_ls_init_U_closed_form = nonneg_ls_init_U


@torch.no_grad()
def nonneg_ls_solve_U_pgd_from_init(
    A: torch.Tensor,
    W: torch.Tensor,
    U_init: torch.Tensor,
    n_iter: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Projected gradient descent NNLS for ``U`` given fixed ``W`` (paper Eq. 5).

    Updates ``U ← ReLU(U − η (U W^T W − A W))`` with step size ``η = 1 / ‖W^T W‖₂``.
    """
    A = A.float()
    W = W.float()
    U = U_init.float().clamp_min(0)

    G = W.t() @ W  # [r, r]
    AW = A @ W  # [N, r]
    # Spectral norm of W^T W gives the Lipschitz constant of the gradient.
    eta = 1.0 / (torch.linalg.matrix_norm(G, ord=2).clamp_min(eps).item())
    for _ in range(n_iter):
        grad = U @ G - AW
        U = (U - eta * grad).clamp_min(0)
    return U


@torch.no_grad()
def nonneg_ls_init_U_hybrid(
    A: torch.Tensor,
    W: torch.Tensor,
    pgd_iters: int = 30,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Closed-form init (Eq. 4) + ``pgd_iters`` of PGD refinement (Eq. 5)."""
    U0 = nonneg_ls_init_U_closed_form(A, W, eps=eps)
    if pgd_iters <= 0:
        return U0
    return nonneg_ls_solve_U_pgd_from_init(A, W, U0, n_iter=pgd_iters, eps=eps)


# ---------------------------------------------------------------------------
# Alignment loss between U (learned coefficients) and C (CLIP semantic activations)
# ---------------------------------------------------------------------------


def alignment_loss_UC(
    U: torch.Tensor,
    C: torch.Tensor,
    mode: str = "soft",
    temperature: float = 0.1,
    entropy_w: float = 0.05,
    ce_targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encourages the learned ``U`` columns to align with CLIP-derived ``C`` columns.

    Three modes supported:
    * ``"soft"``: column-normalize, log-sum-exp matching with optional entropy regularizer.
    * ``"ce"``: cross-entropy of similarity matrix against per-column ``ce_targets``.
    * ``"reg"``: regression — least-squares fit of ``U`` from ``C`` then MSE.
    """
    if C is None:
        return U.new_tensor(0.0)
    Uc = F.normalize(U, p=2, dim=0)
    Cc = F.normalize(C, p=2, dim=0)
    S = Uc.T @ Cc  # [r, M]
    if mode == "soft":
        S_tau = S / temperature
        loss_match = -(temperature * torch.logsumexp(S_tau, dim=1)).mean()
        P = torch.softmax(S_tau, dim=1)
        ent = (-P * (P.clamp_min(1e-8)).log()).sum(dim=1).mean()
        return loss_match + entropy_w * ent
    if mode == "ce":
        if ce_targets is None or len(ce_targets) != S.size(0):
            raise ValueError("'ce' mode requires ce_targets of length r")
        return F.cross_entropy(S / temperature, ce_targets.to(S.device))
    if mode == "reg":
        with torch.no_grad():
            P = torch.linalg.lstsq(C, U).solution  # [M, r]
        U_pred = C @ P
        U_pred = F.normalize(U_pred, p=2, dim=0)
        U_norm = F.normalize(U, p=2, dim=0)
        return F.mse_loss(U_pred, U_norm)
    raise ValueError(f"Unknown align mode: {mode}")


# ---------------------------------------------------------------------------
# LAD trainer: with S fixed, learn W ≥ 0 by PGD on ½ ‖Ā − S W^T‖²_F (paper Eq. 1)
# ---------------------------------------------------------------------------


def train_W_pgd(
    A: torch.Tensor,
    S: torch.Tensor,
    rank: int,
    n_iter: int = 200,
    lr: Optional[float] = None,
    W_init: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, list[float]]:
    """LAD's main optimization: with ``S`` fixed, learn ``W ≥ 0`` (Eq. 1 + Eq. 2).

    Parameters
    ----------
    A : ``[N, C]`` encoder activations (flattened).
    S : ``[N, r]`` fixed CLIP-derived semantic coefficients.
    rank : ``r`` — must match ``S.shape[1]``.
    n_iter : PGD steps.
    lr : step size; if ``None``, set from spectral norm of ``S^T S``.
    W_init : optional ``[C, rank]`` initial basis. If ``None``, NNDSVD is used.

    Returns
    -------
    W : ``[C, rank]`` non-negative basis.
    losses : list of per-iteration objective values.
    """
    if S.shape[1] != rank:
        raise ValueError(f"rank={rank} does not match S.shape[1]={S.shape[1]}")
    A = A.float()
    S = S.float()

    if W_init is None:
        # NNDSVD on A^T (shape [C, N]) returns the first factor with shape [C, r] —
        # exactly the basis W we want for A ≈ S W^T.
        W_seed, _ = nndsvd_init(A.T.contiguous(), rank)
        W = W_seed.contiguous().to(A.device)  # [C, r]
    else:
        W = W_init.float().clamp_min(0).to(A.device)

    StS = S.T @ S  # [r, r]
    StA = S.T @ A  # [r, C]
    if lr is None:
        lr = 1.0 / (torch.linalg.matrix_norm(StS, ord=2).clamp_min(eps).item())

    losses: list[float] = []
    for _ in range(n_iter):
        # gradient of ½ ‖A − S W^T‖²_F w.r.t. W is W S^T S − A^T S
        grad = W @ StS - StA.T
        W = (W - lr * grad).clamp_min(0)
        with torch.no_grad():
            recon = S @ W.T
            losses.append(0.5 * ((recon - A) ** 2).mean().item())
    return W, losses


# ---------------------------------------------------------------------------
# Inference: recover Ŝ for a new image given the trained W
# ---------------------------------------------------------------------------


@torch.no_grad()
def solve_S_hat(
    A: torch.Tensor,
    W: torch.Tensor,
    pgd_iters: int = 30,
) -> torch.Tensor:
    """Inference path: closed-form NNLS init (Eq. 4) + optional PGD refinement (Eq. 5)."""
    return nonneg_ls_init_U_hybrid(A, W, pgd_iters=pgd_iters)


# ---------------------------------------------------------------------------
# Per-batch helper: collect (A, U, y) for a whole dataloader, optionally with
# inner-loop PGD on U + KL-to-teacher (matches FACE / Imagenet_training_with_pipeline.ipynb).
# ---------------------------------------------------------------------------


def collect_A_U_y(
    dataloader,
    g: nn.Module,
    h_2d: nn.Module,
    W: torch.Tensor,
    label_index: int,
    lambda_kl: float = 1e-2,
    optim_steps: int = 0,
    optim_lr: float = 1e-3,
    device: str = "cuda",
):
    """Walk a dataloader, collect flat activations ``A``, coefficients ``U`` and image labels.

    If ``optim_steps > 0`` an inner Adam loop refines each batch's ``U`` to minimize
    ``MSE(A, UW^T) + lambda_kl * KL(logits_A ‖ logits_UW)``. With ``optim_steps == 0`` only
    the closed-form NNLS init is applied (no KL term used).

    Returns
    -------
    A_all : ``[N0, C]`` on CPU.
    U_all : ``[N0, r]`` on CPU.
    y_all : ``[B_total]`` on CPU (constant ``label_index``).
    B_total : number of images.
    L : patches per image (H*W).
    hw : (H, W).
    """
    from tqdm.auto import tqdm

    g.eval()
    W_dev = W.detach().to(device).float()

    A_chunks, U_chunks, y_chunks = [], [], []
    H = Wd = 0

    for images, _ in tqdm(dataloader, desc="collect A/U"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            g_out = g(images)  # new contract: (z_4d, side) for all backbones
            if isinstance(g_out, tuple):
                Z, side = g_out
            else:
                Z, side = g_out, None
            B, _, H, Wd = Z.shape
            A_flat, img_idx, hw = flatten_hw_rowmajor(Z)
            A_bar = unflatten_to_image(A_flat, img_idx, B, hw)
            logits_A = (h_2d(A_bar, side) if side is not None else h_2d(A_bar)).detach()
            U_init = nonneg_ls_init_U(A_flat, W_dev)

        if optim_steps > 0:
            U = nn.Parameter(U_init.clone())
            opt = torch.optim.Adam([U], lr=optim_lr)
            for _ in range(optim_steps):
                opt.zero_grad()
                A_hat = U @ W_dev.T
                A_hat_bar = unflatten_to_image(A_hat, img_idx, B, hw)
                logits_UW = h_2d(A_hat_bar, side) if side is not None else h_2d(A_hat_bar)
                loss = 0.5 * F.mse_loss(A_hat, A_flat) + lambda_kl * kl_logits(logits_A, logits_UW)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    U.data.clamp_(min=0.0)
            U_final = U.data
        else:
            U_final = U_init

        A_chunks.append(A_flat.cpu())
        U_chunks.append(U_final.cpu())
        y_chunks.append(torch.full((B,), label_index, dtype=torch.long))

    A_all = torch.cat(A_chunks, 0)
    U_all = torch.cat(U_chunks, 0)
    y_all = torch.cat(y_chunks, 0)
    B_total = y_all.shape[0]
    L = (A_all.shape[0] // B_total) if B_total else 0
    return A_all, U_all, y_all, B_total, L, (H, Wd)


# ---------------------------------------------------------------------------
# FACE baseline: KL-regularized joint-NMF (replaces the old ``nmfkl.KL_NMF``).
# ---------------------------------------------------------------------------


class KL_NMF:
    """Joint U/W optimization with MSE + λ·KL alignment to teacher logits.

    This is the *FACE* baseline (Bhusal et al., NeurIPS 2025), kept here as a
    comparator. The LAD method does **not** use this class — it fixes ``S``
    (semantic) and learns only ``W`` via :func:`train_W_pgd`.

    Notes
    -----
    The original implementation hard-coded a ResNet-shaped encoder split. To
    extend to other backbones, pass ``g`` / ``h`` / ``h_2d`` callables in
    instead of relying on ``model.fc`` / ``model.children()``.
    """

    def __init__(
        self,
        model: nn.Module,
        imagenet_class: int,
        images_np: np.ndarray,
        rank: int,
        patch_size: int,
        batch_size: int,
        epsilon: float,
        steps: int,
        lambda_val: float,
        learning_rate: float,
        device: str | None = None,
        g: Optional[nn.Module] = None,
        h: Optional[nn.Module] = None,
        h_2d: Optional[nn.Module] = None,
    ) -> None:
        from torchvision import transforms

        self.model = model.eval()
        self.imagenet_class = imagenet_class
        self.images_np = images_np
        self.rank = rank
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.steps = steps
        self.lambda_val = lambda_val
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if g is None or h is None or h_2d is None:
            # Fallback: assume a torchvision ResNet shape.
            self.g = nn.Sequential(*(list(self.model.children())[:-2]))
            self.h = lambda x: self.model.fc(torch.mean(x, (2, 3)))
            self.h_2d = lambda x: self.model.fc(x)
        else:
            self.g, self.h, self.h_2d = g, h, h_2d

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_images(self, images_np):
        from torchvision import transforms

        to_pil = transforms.ToPILImage()
        return torch.stack([self.transform(to_pil(img)) for img in images_np], 0).to(self.device)

    def _batch_inference(self, model_fn, dataset, batch_size: int = 128, resize=None) -> torch.Tensor:
        nb_batches = ceil(len(dataset) / batch_size)
        results = []
        with torch.no_grad():
            for i in range(nb_batches):
                x = dataset[i * batch_size : (i + 1) * batch_size].to(self.device)
                if resize:
                    x = F.interpolate(x, size=resize, mode="bilinear", align_corners=False)
                results.append(model_fn(x).cpu())
        return torch.cat(results)

    def evaluate_accuracy_before_nmf(self) -> float:
        x = self.preprocess_images(self.images_np)
        preds = self._batch_inference(self.model, x, batch_size=4)
        c = torch.argmax(preds, dim=-1)
        return (torch.sum(c == self.imagenet_class) / len(c)).item()

    def compute_activations(self, images_preprocessed):
        strides = int(self.patch_size * 0.80)
        patches = F.unfold(images_preprocessed, kernel_size=self.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size).to(self.device)
        activations = self._batch_inference(self.g, patches, batch_size=self.batch_size).to(self.device)
        if activations.dim() == 4:
            activations_avg = torch.mean(activations, dim=(2, 3))
        else:
            activations_avg = activations
        return patches, activations_avg

    def nmf_kl_pgd(self, patches, activations_avg, input_images):
        """Joint Adam-PGD of (U, W) against MSE + λ·KL."""
        A = activations_avg
        U_init, W_init = nndsvd_init(A, self.rank)
        U = U_init.to(self.device).requires_grad_(True)
        W = W_init.to(self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([U, W], lr=self.learning_rate)
        prev_loss = float("inf")
        mse_losses, kl_losses, total_losses = [], [], []
        original_logits = self._batch_inference(self.model, patches, batch_size=4).to(self.device)

        for _ in range(self.steps):
            optimizer.zero_grad()
            UW = U @ W
            predicted_logits = self.h_2d(UW)
            kl = kl_logits(original_logits, predicted_logits)
            mse = F.mse_loss(UW, A)
            total = mse + self.lambda_val * kl
            total.backward()
            optimizer.step()
            with torch.no_grad():
                U.clamp_(min=0)
                W.clamp_(min=0)
            mse_losses.append(mse.item())
            kl_losses.append(kl.item())
            total_losses.append(total.item())
            if abs(prev_loss - total.item()) < self.epsilon:
                break
            prev_loss = total.item()
        return U.detach(), W.detach(), mse_losses, kl_losses, total_losses


__all__ = [
    # ops
    "flatten_hw_rowmajor",
    "unflatten_to_image",
    "pool_by_image",
    "kl_logits",
    # init
    "nndsvd_init",
    "nonneg_ls_init_U",
    "nonneg_ls_init_U_closed_form",
    "nonneg_ls_init_U_hybrid",
    "nonneg_ls_solve_U_pgd_from_init",
    # alignment
    "alignment_loss_UC",
    # LAD train + inference
    "train_W_pgd",
    "solve_S_hat",
    "collect_A_U_y",
    # FACE baseline
    "KL_NMF",
]
