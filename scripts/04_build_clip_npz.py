"""Step 4: build per-image CLIP red-circle similarity NPZs.

For each filtered image of each class in ``--classes-file``, encode every grid-position
red-circle variant with CLIP, take cosine similarity to that class's concept text features,
and persist the resulting ``[r, h, w]`` tensor as
``data/imagenet_npz/<backbone>/<variant>/correct/<class>/<image_stem>__P_<h>x<w>_r<radius>.npz``.

If launched under ``python -m torch.distributed.run --nproc-per-node=N`` (do **not** use
bare ``torchrun`` on Clariden — it resolves to the uenv binary and spawns workers under
the uenv's Python, which cannot see the venv's site-packages), the work is split across
ranks at the per-class granularity (rank ``i`` handles classes ``i, i+N, i+2N, ...``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # type: ignore  # noqa: E402
from lad.clip_concepts import (
    calculate_importance_and_masks,
    compute_P_for_image,
    encode_text_features,
    load_clip_model,
    load_concepts_txt,
    save_P_npz,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--grid", default="14x14", help="Grid as HxW, e.g. 14x14 or 7x7.")
    p.add_argument("--radius", type=int, default=16)
    p.add_argument("--classes-file", type=Path, default=paths.CLASSES_FILE)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--output-root", type=Path, default=None,
                   help="Default: $LAD_NPZ_ROOT/<backbone>/clip_local_P_ViT-B-16_r<R>_<H>x<W>")
    p.add_argument("--concepts-dir", type=Path, default=paths.CONCEPTS_DIR)
    p.add_argument("--clip-model", default="ViT-B-16")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--batch-size", type=int, default=196)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-per-class", type=int, default=None)
    p.add_argument("--prompt-template", default="a photo of {concept}")
    return p.parse_args()


def parse_grid(s: str) -> tuple[int, int]:
    h, w = s.lower().split("x")
    return int(h), int(w)


def main():
    args = parse_args()
    grid = parse_grid(args.grid)
    if args.output_root is None:
        out_root = paths.NPZ_ROOT / args.backbone / f"clip_local_P_{args.clip_model}_r{args.radius}_{args.grid}"
    else:
        out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if args.device.startswith("cuda") else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if args.device.startswith("cuda") else "cpu"
    else:
        rank, world = 0, 1
        device = args.device

    if rank == 0:
        print(f"Loading CLIP {args.clip_model}/{args.clip_pretrained} on {device} ...")
    clip_model, clip_preprocess, clip_tokenizer, _ = load_clip_model(args.clip_model, pretrained=args.clip_pretrained, device=device)

    classes = [c for c in args.classes_file.read_text().splitlines() if c.strip() and not c.startswith("#")]
    classes = [c.strip() for c in classes]
    classes_for_rank = classes[rank::world]

    backbone_filtered = args.filtered_root / args.backbone / "correct"
    text_cache_dir = out_root / "_text_cache"

    for cls in tqdm(classes_for_rank, desc=f"rank {rank}/{world}"):
        cls_dir = backbone_filtered / cls
        if not cls_dir.exists():
            continue
        concepts_txt = args.concepts_dir / f"{cls}.txt"
        if not concepts_txt.exists():
            print(f"[rank {rank}] missing concepts for {cls}, skipping.")
            continue
        concepts = load_concepts_txt(concepts_txt)
        prompts = [args.prompt_template.format(concept=c) for c in concepts]

        text_features = encode_text_features(
            prompts, clip_model, clip_tokenizer, device,
            cache_path=text_cache_dir / f"{cls}.pt",
        )

        img_paths = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        img_paths.sort()
        if args.limit_per_class:
            img_paths = img_paths[: args.limit_per_class]

        out_dir = out_root / "correct" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"__P_{grid[0]}x{grid[1]}_r{args.radius}.npz"

        for img_path in img_paths:
            out_file = out_dir / f"{img_path.stem}{suffix}"
            if out_file.exists():
                continue
            P, centers, image_hw = compute_P_for_image(
                img_path=img_path,
                text_features=text_features.to(device),
                concepts=prompts,
                model=clip_model,
                device=device,
                grid_size=grid,
                radius=args.radius,
                batch_size=args.batch_size,
            )
            importance, masks = calculate_importance_and_masks(P)
            save_P_npz(
                out_file=out_file,
                P=P,
                concepts=prompts,
                centers=centers,
                grid_size=grid,
                radius=args.radius,
                image_hw=image_hw,
                image_path=img_path,
                importance_scores=importance,
                otsu_masks=masks,
            )

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
