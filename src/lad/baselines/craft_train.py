"""Paper-faithful CRAFT trainer (Fel et al., CVPR 2023).

Mirrors the CRAFT reference evaluation, which uses the upstream ``craft.craft_torch.Craft`` class. CRAFT is unsupervised:

1. **Unfold** each preprocessed 224×224 image into overlapping ``patch_size`` patches
   (default 64×64, stride = ``int(0.8 · patch_size)``).
2. Encode each patch through ``g`` and spatial-mean-pool to ``A: [N_patches, C]``.
3. Run sklearn ``NMF`` on ``A`` → ``U: [N_patches, r]``, ``H: [r, C]``.

We save the basis as ``W = H.T : [C, r]`` (LAD convention).
"""

from __future__ import annotations

from math import ceil
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def _batch_inference(model_fn: Callable, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    n = x.shape[0]
    out = []
    with torch.no_grad():
        for i in range(ceil(n / batch_size)):
            y = model_fn(x[i * batch_size : (i + 1) * batch_size])
            if isinstance(y, tuple):
                y = y[0]  # new g contract returns (z, side_state); baselines drop side
            out.append(y)
    return torch.cat(out, 0)


def unfold_image_patches(
    images_4d: torch.Tensor, patch_size: int = 64, stride_frac: float = 0.8,
) -> torch.Tensor:
    stride = int(patch_size * stride_frac)
    patches = F.unfold(images_4d, kernel_size=patch_size, stride=stride)
    B = images_4d.size(0)
    L = patches.size(-1)
    return patches.transpose(1, 2).contiguous().view(B * L, 3, patch_size, patch_size)


def train_craft_W(
    images_4d: torch.Tensor,
    g,
    rank: int,
    patch_size: int = 64,
    stride_frac: float = 0.8,
    encode_batch_size: int = 64,
    n_iter: int = 500,
    tol: float = 1e-4,
    init: str = "nndsvd",
    seed: int = 42,
) -> Tuple[torch.Tensor, dict]:
    """Paper-faithful CRAFT NMF on patch-level encoder activations.

    Parameters
    ----------
    images_4d : ``[B, 3, 224, 224]`` preprocessed image tensors.
    g : encoder mapping ``[N, 3, P, P] → [N, C, h_p, w_p]`` (or ``[N, C]``).
    rank : number of concepts ``r``.
    patch_size, stride_frac : unfolding hyperparameters.
    n_iter, tol, init, seed : sklearn ``NMF`` hyperparameters.

    Returns
    -------
    W : ``[C, r]`` non-negative basis (transpose of sklearn's ``components_``).
    info : dict with ``n_patches``, ``reconstruction_err_``, ``n_iter_``, ``rank``.
    """
    from sklearn.decomposition import NMF  # imported lazily

    patches = unfold_image_patches(images_4d, patch_size=patch_size, stride_frac=stride_frac)
    patches = patches.to(images_4d.device)
    n_patches = patches.size(0)

    activations = _batch_inference(g, patches, batch_size=encode_batch_size)
    if activations.dim() == 4:
        A = activations.mean(dim=(2, 3))
    else:
        A = activations
    A_np = A.detach().cpu().float().clamp_min(0).numpy()  # NMF requires non-negative

    nmf = NMF(n_components=rank, init=init, max_iter=n_iter, tol=tol, random_state=seed)
    _U = nmf.fit_transform(A_np)  # [N_patches, r] — discarded; recomputed at eval
    H = nmf.components_  # [r, C]
    W_lad = torch.from_numpy(H.T).float()  # [C, r]
    info = {
        "n_patches": int(n_patches),
        "reconstruction_err_": float(nmf.reconstruction_err_),
        "n_iter_": int(nmf.n_iter_),
        "rank": int(rank),
    }
    return W_lad, info


__all__ = ["train_craft_W", "unfold_image_patches"]
