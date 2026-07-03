"""Retinal LAD helper: probe a set of fundus images with a CLIP and learn concept grids.

Shared by ``scripts/retinal/dry_run_retinal.py`` (single CLIP) and ``compare_clips.py``
(multi-CLIP comparison) so the Phase-4/5 logic lives in exactly one place. Pure compute —
no plotting. Mirrors the ImageNet ``scripts/dry_run.py`` math with the retinal pieces swapped
in (ODIR classifier, retinal concepts, fundus probe template, pluggable medical CLIP).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .backbones import load_classifier_checkpoint, make_g_and_h2d
from .clip_concepts import (
    RETINAL_FUNDUS_DOMAIN,
    compute_P_for_image,
    compute_P_tiled,
    encode_text_features,
    load_clip_model,
    load_concepts_txt,
)
from .data import imagenet_transform
from .decomposition import flatten_hw_rowmajor, nonneg_ls_init_U_hybrid, train_W_pgd


def probe_and_learn(
    class_name: str,
    clip: str,
    img_paths: Sequence[Path],
    *,
    checkpoint: Path,
    concepts_dir: Path,
    device: str = "cuda",
    grid: Tuple[int, int] = (14, 14),
    radius: int = 16,
    rank: Optional[int] = None,
    epochs: int = 150,
    clip_checkpoint: Optional[str] = None,
    classifier=None,
    clip_bundle=None,
    input_size: int = 224,
    probe_mode: str = "circle",
    canvas: int = 672,
    window: int = 224,
    prompt_template: Optional[str] = None,
) -> dict:
    """Probe ``img_paths`` with ``clip``, learn the LAD basis W, return per-cell concept grids.

    Returns a dict with: ``S_grid`` ``[B, gh, gw, rank]`` (per-image concept coefficients),
    ``concepts`` (list[str]), ``W`` ``[C, rank]``, ``losses``, ``gh``/``gw``, ``classes``.
    Pass ``classifier=(model, backbone, classes)`` to reuse a loaded checkpoint, and
    ``clip_bundle=(clip_model, clip_tokenizer)`` to reuse a loaded CLIP across many calls.
    """
    if classifier is None:
        model, backbone, classes = load_classifier_checkpoint(checkpoint, device=device)
    else:
        model, backbone, classes = classifier
    g, _ = make_g_and_h2d(model, backbone, device=device)

    concepts = load_concepts_txt(Path(concepts_dir) / f"{class_name}.txt")
    rank = rank or len(concepts)
    h, w = grid

    if clip_bundle is not None:
        clip_model, clip_tok = clip_bundle
    else:
        clip_model, _pre, clip_tok, _ = load_clip_model(clip, device=device, checkpoint=clip_checkpoint)
    template = prompt_template or RETINAL_FUNDUS_DOMAIN.clip_text_template
    prompts = [template.format(c=c) for c in concepts]
    text_feats = encode_text_features(prompts, clip_model, clip_tok, device).to(device)

    tf = imagenet_transform(input_size)  # 448 -> 14x14 encoder grid (vs 7x7 at 224)
    A_chunks: List[torch.Tensor] = []
    S_chunks: List[torch.Tensor] = []
    ghw: Tuple[int, int] = (h, w)
    with torch.inference_mode():
        for ip in img_paths:
            if probe_mode == "tiled":
                P, _centers, _hw = compute_P_tiled(
                    ip, text_feats, prompts, clip_model, device,
                    canvas=canvas, grid_size=(h, w), window=window,
                )
            else:
                P, _centers, _hw = compute_P_for_image(
                    ip, text_feats, prompts, clip_model, device,
                    grid_size=(h, w), radius=radius, batch_size=min(h * w, 256),  # cap VRAM at fine grids
                )
            x = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
            A_flat, _idx, ghw = flatten_hw_rowmajor(g(x)[0])  # g -> (z, side); CNN side is None
            P_dev = P.float().unsqueeze(0).to(device)
            if P_dev.shape[-2:] != ghw:
                P_dev = F.adaptive_avg_pool2d(P_dev, ghw)
            S_flat = P_dev.permute(0, 2, 3, 1).reshape(ghw[0] * ghw[1], len(concepts))
            A_chunks.append(A_flat.cpu())
            S_chunks.append(S_flat.cpu())

    A = torch.cat(A_chunks, 0).to(device).float()
    S = torch.cat(S_chunks, 0).to(device).float()
    W, losses = train_W_pgd(A, S, rank=rank, n_iter=epochs)
    S_hat = nonneg_ls_init_U_hybrid(A, W, pgd_iters=50).cpu()
    gh, gw = ghw
    return {
        "S_grid": S_hat.reshape(len(img_paths), gh, gw, rank),
        "concepts": concepts,
        "W": W.cpu(),
        "losses": losses,
        "gh": gh,
        "gw": gw,
        "classes": classes,
        "backbone": backbone,
    }


def render_cat_figure(img_path, grid_hw, top3, concept_names, *, caption=None, save_path=None):
    """Reference-style figure: [Input Image] -> 3 named concept heatmaps (viridis overlay).

    ``grid_hw`` is one image's ``[gh, gw, rank]`` slice; ``top3`` the concept indices to show.
    Long concept names are wrapped to multiple lines so titles never overflow the column.
    Saves to ``save_path`` if given (caller sets the matplotlib backend / closes the fig).
    Returns the Matplotlib figure.
    """
    import textwrap

    import matplotlib.pyplot as plt

    pil = Image.open(img_path).convert("RGB").resize((224, 224))
    fig = plt.figure(figsize=(13, 3.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.32, 1, 1, 1], wspace=0.06)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(pil); ax0.axis("off")
    ax0.set_title("Input Image", fontsize=13)

    axa = fig.add_subplot(gs[1]); axa.axis("off")
    axa.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", lw=3, color="black"))

    for j, (t, name) in enumerate(zip(top3, concept_names)):
        ax = fig.add_subplot(gs[2 + j])
        heat = grid_hw[..., t].numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        ax.imshow(pil)
        ax.imshow(heat, cmap="viridis", alpha=0.55, extent=[0, 224, 224, 0], interpolation="bilinear")
        ax.axis("off")
        # wrap at spaces only — keep long medical words (e.g. "neovascularization") intact
        wrapped = textwrap.fill(name.title(), width=15, break_long_words=False)
        ax.set_title(f'"{wrapped}"', fontsize=12.5, linespacing=1.05)

    if caption:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color="0.45")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


__all__ = ["probe_and_learn", "render_cat_figure"]
