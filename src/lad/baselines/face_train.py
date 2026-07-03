"""Paper-faithful FACE trainer (Bhusal et al., NeurIPS 2025).

Mirrors the reference implementation in ``D:/Work/XAP CVPR FINALFINAL/FACE/FACE/nmfkl.py``:

1. **Unfold** each preprocessed 224×224 image into overlapping ``patch_size`` patches
   (default 64×64, stride = ``int(0.8 · patch_size)`` = 51 for the default).
2. Encode each patch through the backbone's feature extractor ``g`` and
   spatial-mean-pool to ``A: [N_patches, C]``.
3. Compute per-patch teacher logits ``original_logits = model(patches)``.
4. NNDSVD-init ``U: [N_patches, r]``, ``W: [r, C]``.
5. Adam joint PGD on ``MSE(UW, A) + λ·KL(original_logits, fc(UW))`` with
   ``U.clamp_(0); W.clamp_(0)`` after each step. Optional early stop when
   ``|prev − total| < epsilon``.

The returned ``W`` is in LAD's ``[C, r]`` convention (transpose of FACE's ``[r, C]``)
so the same on-disk artifact is consumed by both the LAD-style eval (``06_eval_metrics``)
and the paper-faithful eval (``09_eval_baselines``).
"""

from __future__ import annotations

from math import ceil
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decomposition import kl_logits, nndsvd_init


def _batch_inference(model_fn: Callable, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Forward ``x`` through ``model_fn`` in chunks of ``batch_size``; outputs concatenated.

    If ``model_fn`` returns a (tensor, side_state) tuple (the new ``g`` contract),
    only the tensor is kept — patch-based baselines do not consume side state.
    """
    n = x.shape[0]
    out = []
    with torch.no_grad():
        for i in range(ceil(n / batch_size)):
            y = model_fn(x[i * batch_size : (i + 1) * batch_size])
            if isinstance(y, tuple):
                y = y[0]
            out.append(y)
    return torch.cat(out, 0)


def unfold_image_patches(
    images_4d: torch.Tensor, patch_size: int = 64, stride_frac: float = 0.8,
) -> torch.Tensor:
    """``F.unfold`` on a [B, 3, H, W] tensor → ``[N_patches, 3, patch, patch]``."""
    stride = int(patch_size * stride_frac)
    patches = F.unfold(images_4d, kernel_size=patch_size, stride=stride)  # [B, 3*P*P, L]
    B = images_4d.size(0)
    L = patches.size(-1)
    patches = patches.transpose(1, 2).contiguous().view(B * L, 3, patch_size, patch_size)
    return patches


def train_face_W(
    images_4d: torch.Tensor,
    model: nn.Module,
    g: nn.Module,
    flat_head: Callable[[torch.Tensor], torch.Tensor],
    rank: int,
    n_iter: int = 22000,
    lr: float = 5e-4,
    lambda_kl: float = 1e-5,
    patch_size: int = 64,
    stride_frac: float = 0.8,
    encode_batch_size: int = 64,
    teacher_batch_size: int = 64,
    eps_early: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, dict]:
    """Paper-faithful FACE training.

    Parameters
    ----------
    images_4d : ``[B, 3, 224, 224]`` preprocessed (ImageNet-mean-normalized) image tensors.
    model : the full classifier, used only to score patches as teacher logits.
    g : encoder; ``g(patches)`` → ``[N_patches, C, h_p, w_p]``.
    flat_head : a callable ``[N, C] → [N, num_classes]`` (e.g.
        ``lambda x: model.fc(x)`` for ResNet, ``model.classifier`` for MobileNet).
    rank : number of concepts ``r``.
    n_iter, lr, lambda_kl, eps_early : optimization hyperparameters (defaults match
        the reference notebook's settings: 22 000 steps, lr 5e-4, λ 1e-5).
    patch_size, stride_frac : unfolding hyperparameters.
        Default ``stride_frac=0.8`` reproduces the reference's ``stride=51`` for ``P=64``.
    encode_batch_size, teacher_batch_size : forward-pass chunk sizes through ``g`` and
        the full ``model``.
    device : default = images' device.

    Returns
    -------
    W : ``[C, r]`` non-negative basis (LAD convention).
    info : dict with ``n_patches``, ``loss_first``, ``loss_last``, ``mse_last``, ``kl_last``,
        ``stopped_at`` (iteration index at which early stop fired, or n_iter).
    """
    if device is None:
        device = str(images_4d.device)
    images_4d = images_4d.to(device)

    patches = unfold_image_patches(images_4d, patch_size=patch_size, stride_frac=stride_frac)
    patches = patches.to(device)
    n_patches = patches.size(0)

    # Encoder activations on patches → [N_patches, C, h_p, w_p] → mean-pool → [N_patches, C]
    activations = _batch_inference(g, patches, batch_size=encode_batch_size)  # 4-D
    if activations.dim() == 4:
        A = activations.mean(dim=(2, 3))
    else:
        A = activations  # already pooled
    A = A.to(device).float()  # [N_patches, C]

    # Teacher logits per patch
    original_logits = _batch_inference(model, patches, batch_size=teacher_batch_size).to(device)

    # NNDSVD init: U [N, r], W [r, C]
    U_init, W_init = nndsvd_init(A, rank)
    U = U_init.to(device).requires_grad_(True)
    W = W_init.to(device).requires_grad_(True)
    opt = torch.optim.Adam([U, W], lr=lr)

    losses, mses, kls = [], [], []
    prev = float("inf")
    stopped_at = n_iter
    for step in range(n_iter):
        opt.zero_grad()
        UW = U @ W  # [N_patches, C]
        logits_hat = flat_head(UW)
        kl = kl_logits(original_logits, logits_hat)
        mse = F.mse_loss(UW, A)
        total = mse + lambda_kl * kl
        total.backward()
        opt.step()
        with torch.no_grad():
            U.clamp_(min=0)
            W.clamp_(min=0)
        v = float(total.item())
        losses.append(v)
        mses.append(float(mse.item()))
        kls.append(float(kl.item()))
        if eps_early > 0 and abs(prev - v) < eps_early:
            stopped_at = step + 1
            break
        prev = v

    info = {
        "n_patches": int(n_patches),
        "stopped_at": int(stopped_at),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "mse_last": mses[-1],
        "kl_last": kls[-1],
    }
    return W.detach().t().contiguous(), info


__all__ = ["train_face_W", "unfold_image_patches"]
