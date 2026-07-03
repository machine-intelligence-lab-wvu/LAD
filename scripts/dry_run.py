"""End-to-end pipeline validation on one class. Run on the laptop to verify the package works.

Default behavior:
  - skip Step 1 (HF download) — pulls images from the local archive instead
  - skip Step 2 (filter) — uses the archive's already-filtered images
  - skip Step 3 (concept gen) — uses the existing concept TXT (verifies API separately)
  - run Step 4 (CLIP red-circle NPZ build) on N images
  - run Step 5 (train W via PGD)
  - emit a visualization PNG of top-k concept heatmaps
  - print metrics (loss, sparsity, reconstruction accuracy)

Override with flags:
  --regen-concepts  re-run gpt-4o-mini for this class (verifies OpenAI live)
  --hf-download     pull images from HuggingFace instead of the archive

Usage (Windows):
  python scripts/dry_run.py --class african_elephant --limit 15
  python scripts/dry_run.py --regen-concepts
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Make repo root importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

from lad.backbones import load_backbone, make_g_and_h2d  # noqa: E402
from lad.clip_concepts import (  # noqa: E402
    calculate_importance_and_masks,
    compute_P_for_image,
    encode_text_features,
    generate_class_vocabulary,
    load_clip_model,
    load_concepts_txt,
    save_P_npz,
)
from lad.data import (  # noqa: E402
    ImagenetImageNpzDataset,
    build_paired_items,
    collate_concept_npz,
    npz_suffix_from_variant,
)
from lad.decomposition import (  # noqa: E402
    flatten_hw_rowmajor,
    nonneg_ls_init_U_hybrid,
    train_W_pgd,
)
from lad.metrics import calculate_gini, compute_sparsity  # noqa: E402


# Default local image pool: the Step-2 filtered/correct ResNet34 folder. Override with
# --archive-root, or pass --hf-download to stream a few images from HuggingFace instead.
ARCHIVE_ROOT = paths.filtered_class_dir("resnet34", "").parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="class_name", default="african_elephant")
    p.add_argument("--limit", type=int, default=12,
                   help="How many images to process. 12 keeps the run under 5 min on CPU.")
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--grid", default="14x14")
    p.add_argument("--radius", type=int, default=16)
    p.add_argument("--regen-concepts", action="store_true",
                   help="Re-run gpt-4o-mini for this class to verify the OpenAI key.")
    p.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT,
                   help="Local image pool: <root>/<class>/*.JPEG (default: Step-2 filtered resnet34).")
    p.add_argument("--hf-download", action="store_true",
                   help="Pull images from HuggingFace instead of the archive.")
    p.add_argument("--out-dir", default=str(REPO / "dry_run_output"))
    return p.parse_args()


def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step_1_get_images(class_name: str, limit: int, hf_download: bool, archive_root: Path) -> Path:
    banner(f"Step 1 — get {limit} images of '{class_name}'")
    out_dir = paths.FILTERED_ROOT / "resnet34" / "correct" / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if hf_download:
        print("Downloading from HuggingFace (streaming)...")
        from datasets import load_dataset
        # Need the int->name mapping to filter to our class
        import csv
        mapping = REPO / "concept_outputs" / "imagenet1k_classes.csv"
        with mapping.open() as f:
            cls_to_idx = {row["simplified"]: int(row["idx"]) for row in csv.DictReader(f)}
        target_idx = cls_to_idx[class_name]
        print(f"  target idx: {target_idx}")
        ds = load_dataset("imagenet-1k", split="train", streaming=True)
        wrote = 0
        t0 = time.perf_counter()
        for ex in ds:
            if ex["label"] != target_idx:
                continue
            ex["image"].save(out_dir / f"hf_{wrote:04d}.JPEG")
            wrote += 1
            if wrote >= limit:
                break
            if wrote % 5 == 0:
                print(f"  wrote {wrote}/{limit} (elapsed {time.perf_counter()-t0:.1f}s)")
        print(f"  done — {wrote} images in {time.perf_counter()-t0:.1f}s")
    else:
        if not archive_root.exists():
            raise RuntimeError(
                f"Archive not found at {archive_root}. Pass --archive-root or use --hf-download."
            )
        archive_dir = archive_root / class_name
        if not archive_dir.exists():
            raise RuntimeError(f"No archive folder for {class_name!r} at {archive_dir}")
        candidates = sorted(archive_dir.glob("*.jpg"))[:limit]
        for p in candidates:
            shutil.copy2(p, out_dir / p.name)
        print(f"  copied {len(candidates)} images from {archive_dir}")
    return out_dir


def step_3_concepts(class_name: str, regen: bool, device: str) -> List[str]:
    banner(f"Step 3 — concept vocabulary for '{class_name}'")
    txt_path = paths.CONCEPTS_DIR / f"{class_name}.txt"
    if txt_path.exists() and not regen:
        concepts = load_concepts_txt(txt_path)
        print(f"  using existing TXT: {txt_path}")
        print(f"  loaded {len(concepts)} concepts")
        for c in concepts[:3]:
            print(f"    - {c}")
        if len(concepts) > 3:
            print(f"    ... and {len(concepts)-3} more")
        return concepts

    print("  re-running gpt-4o-mini (verifies OPENAI_API_KEY)...")
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set — set it first or skip --regen-concepts.")
    clip_model, clip_pre, clip_tok, _ = load_clip_model("ViT-B-16", "laion2b_s34b_b88k", device=device)
    cls_image_dir = paths.FILTERED_ROOT / "resnet34" / "correct" / class_name
    vocab, stats = generate_class_vocabulary(
        class_name,
        max_rounds=5,
        clip_model=clip_model,
        clip_preprocess=clip_pre,
        clip_tokenizer=clip_tok,
        clip_device=device,
        class_image_dir=cls_image_dir,
    )
    txt_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    print(f"  generated {len(vocab)} concepts in {stats.rounds} rounds; wrote {txt_path}")
    return vocab


def step_4_build_npz(class_name: str, concepts: List[str], grid: str, radius: int, device: str):
    banner(f"Step 4 — build CLIP red-circle NPZs (grid={grid}, radius={radius})")
    h, w = (int(x) for x in grid.split("x"))
    images_dir = paths.FILTERED_ROOT / "resnet34" / "correct" / class_name
    out_dir = paths.NPZ_ROOT / "resnet34" / f"clip_local_P_ViT-B-16_r{radius}_{grid}" / "correct" / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"__P_{h}x{w}_r{radius}.npz"

    print(f"  loading CLIP ViT-B-16 ({device}) ...")
    clip_model, clip_pre, clip_tok, _ = load_clip_model("ViT-B-16", "laion2b_s34b_b88k", device=device)
    prompts = [f"a photo of {c}" for c in concepts]
    text_feats = encode_text_features(prompts, clip_model, clip_tok, device)

    img_paths = sorted(images_dir.glob("*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    print(f"  building NPZs for {len(img_paths)} images ...")
    t0 = time.perf_counter()
    for i, ip in enumerate(img_paths, 1):
        out_file = out_dir / f"{ip.stem}{suffix}"
        if out_file.exists():
            continue
        P, centers, image_hw = compute_P_for_image(
            img_path=ip,
            text_features=text_feats.to(device),
            concepts=prompts,
            model=clip_model,
            device=device,
            grid_size=(h, w),
            radius=radius,
            batch_size=h * w,  # all grid positions in one batch
        )
        importance, masks = calculate_importance_and_masks(P)
        save_P_npz(
            out_file=out_file,
            P=P,
            concepts=prompts,
            centers=centers,
            grid_size=(h, w),
            radius=radius,
            image_hw=image_hw,
            image_path=ip,
            importance_scores=importance,
            otsu_masks=masks,
        )
        if i % 5 == 0 or i == len(img_paths):
            print(f"    {i}/{len(img_paths)} done ({(time.perf_counter()-t0)/i:.1f}s/img)")
    print(f"  total NPZ build time: {time.perf_counter()-t0:.1f}s")
    return out_dir


def step_5_train(class_name: str, npz_variant: str, rank: int, epochs: int, device: str):
    banner(f"Step 5 — train W via PGD (rank={rank}, epochs={epochs})")

    images_root = paths.FILTERED_ROOT / "resnet34" / "correct"
    npz_root = paths.NPZ_ROOT / "resnet34" / f"clip_local_P_{npz_variant}" / "correct"
    suffix = npz_suffix_from_variant(npz_variant)
    classes, _, paired, report = build_paired_items(images_root, npz_root, suffix, classes=[class_name])
    print("  pairing report:", report)
    if not paired:
        raise RuntimeError("No paired items.")

    from torch.utils.data import DataLoader
    ds = ImagenetImageNpzDataset(paired)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                        collate_fn=collate_concept_npz)

    model = load_backbone("resnet34", device=device)
    g, h_2d = make_g_and_h2d(model, "resnet34", device=device)

    A_chunks, S_chunks = [], []
    print("  collecting A and S ...")
    with torch.inference_mode():
        for imgs, ys, P, *_rest in loader:
            imgs = imgs.to(device)
            Z, _side = g(imgs)
            A_flat, _idx, hw = flatten_hw_rowmajor(Z)
            B = imgs.size(0)
            r = P.shape[1]
            P_dev = P.float().to(device)
            if P_dev.shape[-2:] != hw:
                P_dev = F.adaptive_avg_pool2d(P_dev, hw)
            S_flat = P_dev.permute(0, 2, 3, 1).reshape(B * hw[0] * hw[1], r)
            A_chunks.append(A_flat.cpu())
            S_chunks.append(S_flat.cpu())

    A = torch.cat(A_chunks, 0).to(device).float()
    S = torch.cat(S_chunks, 0).to(device).float()
    print(f"  A: {tuple(A.shape)}, S: {tuple(S.shape)}, rank: {rank}")

    t0 = time.perf_counter()
    W, losses = train_W_pgd(A, S, rank=rank, n_iter=epochs)
    elapsed = time.perf_counter() - t0
    print(f"  trained in {elapsed:.1f}s; loss[0]={losses[0]:.4e}  loss[-1]={losses[-1]:.4e}")
    print(f"  W min/max/mean: {W.min().item():.4f}/{W.max().item():.4f}/{W.mean().item():.4f}")
    print(f"  W zeros fraction: {(W == 0).float().mean().item():.3f}")

    # Reconstruction quality + recon accuracy
    with torch.inference_mode():
        S_hat = nonneg_ls_init_U_hybrid(A, W, pgd_iters=30)
        A_hat = S_hat @ W.T
        mse = F.mse_loss(A_hat, A).item()
        # Reshape A and A_hat back to spatial and run h_2d
        B_total = A.shape[0] // (hw[0] * hw[1])
        A_orig_4d = A.reshape(B_total, hw[0], hw[1], -1).permute(0, 3, 1, 2)
        A_hat_4d = A_hat.reshape(B_total, hw[0], hw[1], -1).permute(0, 3, 1, 2)
        logits_orig = h_2d(A_orig_4d)
        logits_hat = h_2d(A_hat_4d)
        # Get the canonical class idx
        import csv as _csv
        with (REPO / "concept_outputs" / "imagenet1k_classes.csv").open() as f:
            cls_to_idx = {row["simplified"]: int(row["idx"]) for row in _csv.DictReader(f)}
        target_idx = cls_to_idx[class_name]
        acc_orig = (logits_orig.argmax(-1) == target_idx).float().mean().item()
        acc_recon = (logits_hat.argmax(-1) == target_idx).float().mean().item()
        sparsity_S = compute_sparsity(S_hat.reshape(B_total, hw[0]*hw[1], rank).mean(dim=1))
    print(f"  recon MSE: {mse:.4e}")
    print(f"  acc original: {acc_orig:.3f}   acc reconstruction: {acc_recon:.3f}")
    print(f"  S_hat sparsity (per image): {sparsity_S:.3f}")
    return (W.cpu(), S_hat.cpu(),
            {"loss_first": losses[0], "loss_last": losses[-1],
             "mse": mse, "acc_orig": acc_orig, "acc_recon": acc_recon,
             "sparsity": sparsity_S, "elapsed_s": elapsed},
            B_total, hw)


def step_6_visualize(class_name: str, npz_variant: str, W: torch.Tensor, S_hat: torch.Tensor,
                     concepts: List[str], grid: str, out_dir: Path, B: int, hw: tuple,
                     top_k: int = 5, n_imgs: int = 4):
    banner("Step 6 — visualize top concepts")
    import matplotlib.pyplot as plt
    h, w = hw  # use the encoder's actual spatial size, not the CLIP probe grid
    rank = W.shape[1]
    print(f"  encoder spatial: {h}x{w}, B={B}, S_hat shape: {tuple(S_hat.shape)}")

    images_dir = paths.FILTERED_ROOT / "resnet34" / "correct" / class_name
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in
                        {".jpg", ".jpeg", ".png"}])[:min(n_imgs, B)]

    # S_hat is [B*h*w, r] row-major; reshape per-image
    S_grid_all = S_hat.reshape(B, h, w, rank)
    n_imgs = min(n_imgs, B, len(img_paths))

    fig, axes = plt.subplots(n_imgs, top_k + 1, figsize=(2.4 * (top_k + 1), 2.4 * n_imgs))
    if n_imgs == 1:
        axes = axes[None, :]

    for r_idx in range(n_imgs):
        S_grid = S_grid_all[r_idx]  # [h, w, r]
        importance = S_grid.sum(dim=(0, 1))
        top_idx = importance.topk(top_k).indices.tolist()

        pil = Image.open(img_paths[r_idx]).convert("RGB").resize((224, 224))
        axes[r_idx, 0].imshow(pil)
        axes[r_idx, 0].axis("off")
        if r_idx == 0:
            axes[r_idx, 0].set_title("input", fontsize=9)

        for c_idx, k in enumerate(top_idx, 1):
            heat = S_grid[..., k].numpy()
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            axes[r_idx, c_idx].imshow(pil, alpha=0.6)
            axes[r_idx, c_idx].imshow(
                heat, cmap="jet", alpha=0.5,
                extent=[0, 224, 224, 0], interpolation="bilinear",
            )
            axes[r_idx, c_idx].axis("off")
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(concepts[k][:18], fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{class_name} — top {top_k} concepts per image", fontsize=11)
    fig.tight_layout()
    out_file = out_dir / f"dry_run_{class_name}_top{top_k}.png"
    fig.savefig(out_file, dpi=140, bbox_inches="tight")
    print(f"  saved {out_file}")
    plt.close(fig)
    return out_file


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | class: {args.class_name} | limit: {args.limit}")

    out_dir = Path(args.out_dir)

    step_1_get_images(args.class_name, args.limit, args.hf_download, args.archive_root)
    concepts = step_3_concepts(args.class_name, args.regen_concepts, device)
    step_4_build_npz(args.class_name, concepts, args.grid, args.radius, device)

    npz_variant = f"ViT-B-16_r{args.radius}_{args.grid}"
    W, S_hat, metrics, B, hw = step_5_train(args.class_name, npz_variant, args.rank, args.epochs, device)

    out_file = step_6_visualize(args.class_name, npz_variant, W, S_hat, concepts,
                                args.grid, out_dir, B, hw)

    banner("DONE")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  visualization: {out_file}")


if __name__ == "__main__":
    main()
