"""Step 7: paper-faithful FACE training per (backbone, class, fold).

Replicates the FACE reference implementation (Bhusal et al., NeurIPS 2025):

* unfold each 224×224 image into ``patch_size`` patches with 80% overlap,
* encode each patch through the backbone, mean-pool → ``A: [N_patches, C]``,
* per-patch teacher logits via the full classifier,
* NNDSVD-init + Adam joint PGD on ``MSE(UW, A) + λ·KL(orig, fc(UW))``,
* save the basis as ``W: [C, r]`` (LAD convention) so any of the eval scripts
  (``06_eval_metrics`` or ``09_eval_baselines``) can pick it up.

Default output:
    outputs/grid_face/<backbone>/<npz_variant>/fold_<fold>/<class>/W_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
import paths  # type: ignore  # noqa: E402
from lad.backbones import load_backbone, make_g_and_h2d
from lad.baselines.face_train import train_face_W
from lad.data import (
    ImagenetImageNpzDataset,
    build_paired_items,
    collate_concept_npz,
    npz_suffix_from_variant,
    per_class_kfold,
)


def _flat_head_for(model, backbone: str):
    """Return a callable ``[N, C] -> [N, num_classes]`` matching the reference's ``h_2d``.

    For ResNet, this is ``model.fc``. For MobileNet-V2 / ConvNeXt, it's ``model.classifier``
    (which expects flat ``[N, C]`` input as per torchvision's spec for those heads).
    """
    backbone = backbone.lower()
    if backbone.startswith("resnet"):
        return model.fc
    if backbone == "mobilenet_v2":
        return model.classifier  # Sequential(Dropout, Linear) accepting [N, C]
    if backbone.startswith("convnext"):  # tiny / small / small_ham10000
        # ConvNeXt classifier expects [N, C, 1, 1] (LayerNorm2d + Flatten + Linear).
        cls = model.classifier
        return lambda x: cls(x.unsqueeze(-1).unsqueeze(-1))
    raise NotImplementedError(f"flat head not defined for {backbone!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--class", dest="class_name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npz-variant", default="ViT-B-16_r16_14x14",
                   help="Used only for paired-items lookup and the output dir name.")
    p.add_argument("--rank", type=int, default=25)
    # Reference defaults: steps=22000, lr=5e-4, lambda_val=1e-5, epsilon=1e-3, patch_size=64
    p.add_argument("--epochs", type=int, default=22000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lambda-kl", type=float, default=1e-5)
    p.add_argument("--epsilon", type=float, default=1e-3, help="Early-stop tolerance on total loss.")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--stride-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Image-loading batch size (training data is concatenated).")
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--teacher-batch-size", type=int, default=32)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--npz-root", type=Path, default=paths.NPZ_ROOT)
    p.add_argument("--output-root", type=Path, default=paths.OUTPUT_ROOT / "grid_face")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    images_root = args.filtered_root / args.backbone / "correct"
    npz_root = args.npz_root / args.backbone / f"clip_local_P_{args.npz_variant}" / "correct"
    if not (images_root / args.class_name).exists():
        raise FileNotFoundError(f"No filtered images for {args.class_name!r} under {images_root}")
    if not (npz_root / args.class_name).exists():
        raise FileNotFoundError(f"No NPZ files for {args.class_name!r} under {npz_root}")

    suffix = npz_suffix_from_variant(args.npz_variant)
    _classes, _c2i, paired, report = build_paired_items(
        images_root, npz_root, suffix, classes=[args.class_name],
    )
    print("paired-items report:", report)
    if not paired:
        raise RuntimeError(f"No paired items for class {args.class_name!r}")
    train_idx, val_idx = per_class_kfold(paired, args.folds, args.fold,
                                         val_ratio=args.val_ratio, seed=args.seed)

    full_ds = ImagenetImageNpzDataset(paired)
    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"), drop_last=False, collate_fn=collate_concept_npz,
    )

    print(f"Loading backbone {args.backbone} on {device} ...")
    model = load_backbone(args.backbone, device=device)
    g, _h_2d = make_g_and_h2d(model, args.backbone, device=device)
    flat_head = _flat_head_for(model, args.backbone)

    print("Collecting training images (224x224) ...")
    img_chunks = []
    for imgs, *_rest in train_loader:
        img_chunks.append(imgs)
    images_4d = torch.cat(img_chunks, 0).to(device)
    print(f"images_4d: {tuple(images_4d.shape)}   patch_size={args.patch_size}   "
          f"stride={int(args.patch_size * args.stride_frac)}")

    print("Training paper-faithful FACE (patches + KL) ...")
    t0 = time.perf_counter()
    W, info = train_face_W(
        images_4d=images_4d,
        model=model,
        g=g,
        flat_head=flat_head,
        rank=args.rank,
        n_iter=args.epochs,
        lr=args.lr,
        lambda_kl=args.lambda_kl,
        patch_size=args.patch_size,
        stride_frac=args.stride_frac,
        encode_batch_size=args.encode_batch_size,
        teacher_batch_size=args.teacher_batch_size,
        eps_early=args.epsilon,
        device=device,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.1f}s. n_patches={info['n_patches']}  "
          f"stopped_at={info['stopped_at']}  loss {info['loss_first']:.4f} -> {info['loss_last']:.4f}")

    out_dir = args.output_root / args.backbone / args.npz_variant / f"fold_{args.fold}" / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(W.detach().cpu(), out_dir / "W_final.pt")
    history = {
        "method": "face_paperfaithful",
        "backbone": args.backbone, "class": args.class_name, "fold": args.fold,
        "npz_variant": args.npz_variant, "rank": args.rank,
        "epochs": args.epochs, "lr": args.lr, "lambda_kl": args.lambda_kl,
        "patch_size": args.patch_size, "stride_frac": args.stride_frac,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "elapsed_s": elapsed, **info,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Wrote {out_dir / 'W_final.pt'}")


if __name__ == "__main__":
    main()
