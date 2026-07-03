"""Phases 4+5 end-to-end on one retinal class -> the concept-map figure (the "cat result").

Mirrors scripts/dry_run.py but swaps in the retinal pieces:
  * classifier  = Phase-1 ODIR checkpoint (load_classifier_checkpoint), not an ImageNet model
  * concepts    = concept_outputs/retinal/<class>.txt (Phase 3)
  * CLIP        = a medical/clip registry model (default biomedclip), fundus probe template
  * images      = data/odir/filtered/<backbone>/correct/<class>/ (Phase 2, correctly classified)

For N images it builds the red-circle CLIP similarity tensor P=[r,14,14], pools it to the
encoder's 7x7 grid as S, learns the non-negative basis W via PGD (LAD Eq. 1), recovers the
per-cell concept coefficients S_hat (Eq. 4), and renders top-k concept heatmaps per image.

Usage (lad conda env):
  python scripts/retinal/dry_run_retinal.py --class diabetic_retinopathy --clip biomedclip --limit 24
  python scripts/retinal/dry_run_retinal.py --class amd --clip laion --limit 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402
from lad.backbones import load_classifier_checkpoint, make_g_and_h2d  # noqa: E402
from lad.clip_concepts import (  # noqa: E402
    RETINAL_FUNDUS_DOMAIN,
    compute_P_for_image,
    encode_text_features,
    load_clip_model,
    load_concepts_txt,
)
from lad.decomposition import (  # noqa: E402
    flatten_hw_rowmajor,
    nonneg_ls_init_U_hybrid,
    train_W_pgd,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class", dest="class_name", default="diabetic_retinopathy")
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--clip", default="biomedclip", help="CLIP name (biomedclip/laion/openai/fairclip).")
    p.add_argument("--clip-checkpoint", default=None, help="FairCLIP .pth path (required for --clip fairclip).")
    p.add_argument("--limit", type=int, default=24, help="Images to probe (each ~196 CLIP encodes).")
    p.add_argument("--rank", type=int, default=None, help="NMF rank; default = number of concepts.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--grid", default="14x14")
    p.add_argument("--radius", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-imgs", type=int, default=4)
    p.add_argument("--layout", choices=["topk", "gallery"], default="topk",
                   help="topk = rows x top-k concepts; gallery = grid of images each with its top-1 concept.")
    p.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "retinal")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def banner(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def main():
    args = parse_args()
    dev = args.device
    h, w = (int(x) for x in args.grid.split("x"))
    ckpt = args.checkpoint or (paths.MODELS_DIR / f"best_{args.backbone}_odir.pt")

    banner(f"{args.class_name} | clip={args.clip} | {args.limit} imgs | device={dev}")
    model, backbone, classes = load_classifier_checkpoint(ckpt, device=dev)
    g, h_2d = make_g_and_h2d(model, backbone, device=dev)
    cls_idx = classes.index(args.class_name)
    print(f"classifier classes={classes}  target='{args.class_name}' (idx {cls_idx})")

    concepts = load_concepts_txt(paths.RETINAL_CONCEPTS_DIR / f"{args.class_name}.txt")
    rank = args.rank or len(concepts)
    print(f"{len(concepts)} concepts, rank={rank}")

    img_dir = paths.RETINAL_FILTERED_ROOT / backbone / "correct" / args.class_name
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not img_paths:
        raise SystemExit(f"No filtered images at {img_dir} — run 02_filter_correct.py first.")
    img_paths = img_paths[: args.limit]

    # CLIP text features for the fundus probe template (load_clip_model resolves registry names)
    clip_model, _clip_pre, clip_tok, _ = load_clip_model(args.clip, device=dev, checkpoint=args.clip_checkpoint)
    prompts = [RETINAL_FUNDUS_DOMAIN.clip_text_template.format(c=c) for c in concepts]
    text_feats = encode_text_features(prompts, clip_model, clip_tok, dev).to(dev)

    banner(f"Step 4 — red-circle CLIP probing ({len(img_paths)} imgs, grid {h}x{w})")
    A_chunks, S_chunks = [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, ip in enumerate(img_paths, 1):
            # CLIP concept similarity tensor P=[r,h,w]
            P, _centers, _hw = compute_P_for_image(
                ip, text_feats, prompts, clip_model, dev,
                grid_size=(h, w), radius=args.radius, batch_size=h * w,
            )
            # encoder activations A on the SAME image
            img = Image.open(ip).convert("RGB")
            from lad.data import imagenet_transform
            x = imagenet_transform()(img).unsqueeze(0).to(dev)
            Z, _ = g(x)                                # g -> (z, side); [1, C, gh, gw]
            A_flat, _idx, ghw = flatten_hw_rowmajor(Z)  # [gh*gw, C]
            # pool CLIP grid -> encoder grid and flatten row-major to match A
            P_dev = P.float().unsqueeze(0).to(dev)      # [1, r, h, w]
            if P_dev.shape[-2:] != ghw:
                P_dev = F.adaptive_avg_pool2d(P_dev, ghw)
            S_flat = P_dev.permute(0, 2, 3, 1).reshape(ghw[0] * ghw[1], len(concepts))
            A_chunks.append(A_flat.cpu())
            S_chunks.append(S_flat.cpu())
            if i % 5 == 0 or i == len(img_paths):
                print(f"  {i}/{len(img_paths)}  ({(time.perf_counter()-t0)/i:.2f}s/img)")

    A = torch.cat(A_chunks, 0).to(dev).float()
    S = torch.cat(S_chunks, 0).to(dev).float()
    B = len(img_paths)
    gh, gw = ghw
    print(f"A={tuple(A.shape)}  S={tuple(S.shape)}  (B={B}, grid={gh}x{gw})")

    banner(f"Step 5 — learn basis W via PGD (rank={rank}, epochs={args.epochs})")
    W, losses = train_W_pgd(A, S, rank=rank, n_iter=args.epochs)
    print(f"loss {losses[0]:.3e} -> {losses[-1]:.3e}   W>=0: {bool((W>=0).all())}  "
          f"sparsity(W==0): {(W==0).float().mean().item():.2f}")

    S_hat = nonneg_ls_init_U_hybrid(A, W, pgd_iters=50).cpu()   # [B*gh*gw, rank]
    S_grid = S_hat.reshape(B, gh, gw, rank)

    banner(f"Step 6 — render top-{args.top_k} concept heatmaps")
    out_file = render(args, concepts, S_grid, img_paths, gh, gw)
    print(f"\nDONE -> {out_file}")


def render(args, concepts, S_grid, img_paths, gh, gw):
    if args.layout == "gallery":
        return render_gallery(args, concepts, S_grid, img_paths)
    return render_topk(args, concepts, S_grid, img_paths)


def render_gallery(args, concepts, S_grid, img_paths):
    """Grid of fundus images, each overlaid with ITS single most salient LAD concept."""
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(args.n_imgs, S_grid.shape[0], len(img_paths))
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.9 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        grid = S_grid[i]
        idx = int(grid.sum(dim=(0, 1)).argmax())
        heat = grid[..., idx].numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        pil = Image.open(img_paths[i]).convert("RGB").resize((224, 224))
        ax.imshow(pil, alpha=0.6)
        ax.imshow(heat, cmap="jet", alpha=0.5, extent=[0, 224, 224, 0], interpolation="bilinear")
        ax.set_title(concepts[idx][:26], fontsize=9)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{args.class_name} — top LAD concept per fundus  (CLIP={args.clip})", fontsize=12)
    fig.tight_layout()
    out_file = args.out_dir / f"retinal_{args.class_name}_{args.backbone}_{args.clip}_gallery.png"
    fig.savefig(out_file, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_file


def render_topk(args, concepts, S_grid, img_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(args.n_imgs, S_grid.shape[0], len(img_paths))
    k = args.top_k
    fig, axes = plt.subplots(n, k + 1, figsize=(2.3 * (k + 1), 2.3 * n))
    if n == 1:
        axes = axes[None, :]

    for r in range(n):
        grid = S_grid[r]                      # [gh, gw, rank]
        imp = grid.sum(dim=(0, 1))
        top = imp.topk(k).indices.tolist()
        pil = Image.open(img_paths[r]).convert("RGB").resize((224, 224))
        axes[r, 0].imshow(pil); axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title("fundus", fontsize=9)
        for c, idx in enumerate(top, 1):
            heat = grid[..., idx].numpy()
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            axes[r, c].imshow(pil, alpha=0.55)
            axes[r, c].imshow(heat, cmap="jet", alpha=0.5, extent=[0, 224, 224, 0],
                              interpolation="bilinear")  # smooth 7x7 -> 224 like the cat figure
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(concepts[idx][:20], fontsize=8)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{args.class_name} — top-{k} LAD concepts per image  (CLIP={args.clip})", fontsize=11)
    fig.tight_layout()
    out_file = args.out_dir / f"retinal_{args.class_name}_{args.backbone}_{args.clip}_top{k}.png"
    fig.savefig(out_file, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_file


if __name__ == "__main__":
    main()
