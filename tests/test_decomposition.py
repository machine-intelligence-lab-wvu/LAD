"""Unit tests for the LAD core decomposition kernel.

These tests run on a small synthetic problem so they finish in seconds and don't require
GPU, ImageNet, or any data on disk. They're the gate for "did the refactor break math".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lad.decomposition import (
    flatten_hw_rowmajor,
    nndsvd_init,
    nonneg_ls_init_U,
    nonneg_ls_init_U_hybrid,
    pool_by_image,
    solve_S_hat,
    train_W_pgd,
    unflatten_to_image,
)


def _make_synthetic(N=200, C=64, rank=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    S_true = torch.rand(N, rank, generator=g)
    W_true = torch.rand(C, rank, generator=g)
    A = S_true @ W_true.T + 0.01 * torch.randn(N, C, generator=g)
    return A, S_true, W_true


def test_flatten_unflatten_roundtrip():
    """Row-major flatten + scatter unflatten is the identity."""
    z = torch.arange(2 * 3 * 4 * 5, dtype=torch.float).reshape(2, 3, 4, 5)
    A, idx, hw = flatten_hw_rowmajor(z)
    z2 = unflatten_to_image(A, idx, B=2, hw=hw)
    assert torch.equal(z, z2)


def test_pool_by_image_matches_torch_mean():
    z = torch.randn(3, 5, 4, 4)
    A, idx, _ = flatten_hw_rowmajor(z)
    pooled = pool_by_image(A, idx, B=3)
    expected = z.mean(dim=(2, 3))
    torch.testing.assert_close(pooled, expected, rtol=1e-5, atol=1e-5)


def test_nonneg_ls_init_is_nonneg():
    A, _S, W = _make_synthetic()
    U0 = nonneg_ls_init_U(A, W)
    assert U0.shape == (A.shape[0], W.shape[1])
    assert torch.all(U0 >= 0)


def test_nonneg_ls_hybrid_reduces_error():
    """PGD refinement after closed-form init reduces reconstruction MSE."""
    A, _S, W = _make_synthetic()
    U0 = nonneg_ls_init_U(A, W)
    U1 = nonneg_ls_init_U_hybrid(A, W, pgd_iters=50)
    err0 = ((U0 @ W.T - A) ** 2).mean().item()
    err1 = ((U1 @ W.T - A) ** 2).mean().item()
    assert err1 <= err0 + 1e-6
    assert torch.all(U1 >= 0)


def test_nndsvd_init_shapes_and_nonneg():
    A, _, _ = _make_synthetic()
    U0, W0 = nndsvd_init(A, r=8)
    assert U0.shape == (A.shape[0], 8)
    assert W0.shape == (8, A.shape[1])
    assert torch.all(U0 >= 0)
    assert torch.all(W0 >= 0)


def test_train_W_pgd_decreases_loss_and_stays_nonneg():
    """LAD's main loss must decrease and W must stay non-negative."""
    A, S_true, _W_true = _make_synthetic(N=300, C=48, rank=6)
    W, losses = train_W_pgd(A, S_true, rank=6, n_iter=80)
    assert W.shape == (48, 6)
    assert torch.all(W >= 0)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3e} -> {losses[-1]:.3e}"
    # We expect the final loss to be much lower than the initial (synthetic, no noise but small).
    assert losses[-1] < 0.5 * losses[0]


def test_solve_S_hat_recovers_S_true_on_clean_data():
    """With clean A = S_true @ W_true^T and exact W, solve_S_hat should recover S_true closely."""
    g = torch.Generator().manual_seed(7)
    rank = 5
    S_true = torch.rand(80, rank, generator=g)
    W_true = torch.rand(32, rank, generator=g)
    A = S_true @ W_true.T  # noise-free
    S_hat = solve_S_hat(A, W_true, pgd_iters=200)
    # On noise-free synthetic data the recovery should be tight.
    rel = ((S_hat - S_true).norm() / S_true.norm()).item()
    assert rel < 5e-2, f"relative recovery error {rel:.3e} too large"


@pytest.mark.parametrize("rank", [3, 6, 12])
def test_train_W_pgd_rank_consistency(rank):
    """train_W_pgd respects the rank argument."""
    A, S, _ = _make_synthetic(N=120, C=24, rank=rank)
    W, _ = train_W_pgd(A, S, rank=rank, n_iter=20)
    assert W.shape == (24, rank)
