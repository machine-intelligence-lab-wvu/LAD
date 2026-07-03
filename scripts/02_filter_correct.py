"""Step 2: walk an ImageNet split and write a manifest of correctly-classified images per class.

For each (backbone, image), we run the pretrained classifier and record only those whose
top-1 prediction matches the ground-truth class. The manifest layout matches the original
``Dataset/Imagenet_filtered/<backbone>/correct/<class_name>/`` convention:

  - by default, we write **symlinks** so we don't duplicate ImageNet on disk.
  - pass ``--copy`` to copy files instead (use only if symlinks aren't possible).

Usage:
  python scripts/02_filter_correct.py --backbone resnet34 \
      --classes-file concept_outputs/classes_500.txt \
      --split train
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

# Make `import paths` work regardless of cwd (paths.py is in the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # type: ignore  # noqa: E402
from lad.backbones import load_backbone
from lad.data import imagenet_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True,
                   choices=["resnet18", "resnet34", "resnet50", "mobilenet_v2",
                            "convnext_tiny", "resnet50_places365",
                            "convnext_small_ham10000", "convnext_small_ham10000_448"])
    p.add_argument("--img-size", type=int, default=224,
                   help="Classifier input resolution (448 for the _448 fine-tune).")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--classes-file", type=Path, default=paths.CLASSES_FILE,
                   help="Newline-separated class names (or wnids) to keep. None = all.")
    p.add_argument("--imagenet-root", type=Path, default=paths.IMAGENET_ROOT,
                   help="Root containing the <split> subfolder with class subdirs.")
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT)
    p.add_argument("--mapping-csv", type=Path,
                   default=Path(__file__).resolve().parent.parent
                           / "concept_outputs" / "imagenet1k_classes.csv",
                   help="CSV with columns idx,raw,simplified mapping folder names to "
                        "the global class index the backbone outputs. For Places365, pass "
                        "concept_outputs/places365_classes.csv (emitted by places365_setup.py).")
    p.add_argument("--no-split-subdir", action="store_true",
                   help="If set, treat --imagenet-root *itself* as the per-class image dir "
                        "(skip the train/validation subdirectory). Useful for the Places365 "
                        "val-only layout produced by places365_setup.py.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_classes(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def main():
    args = parse_args()
    device = args.device

    split_dir = args.imagenet_root if args.no_split_subdir else (args.imagenet_root / args.split)
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing image root: {split_dir}")
    out_root = args.filtered_root / args.backbone / "correct"
    out_root.mkdir(parents=True, exist_ok=True)

    keep = load_classes(args.classes_file)

    print(f"Loading {args.backbone} on {device}...")
    model = load_backbone(args.backbone, device=device)

    ds = ImageFolder(str(split_dir), transform=imagenet_transform(args.img_size))
    if keep is not None:
        # ImageFolder labels are sorted class names → integer labels matching the alphabetical order
        # of folders inside split_dir. ImageNet wnids in our `keep` set must match folder names.
        keep_idx = {ds.class_to_idx[c] for c in keep if c in ds.class_to_idx}
        ds.samples = [(p, y) for (p, y) in ds.samples if y in keep_idx]
        ds.targets = [y for (_p, y) in ds.samples]
        print(f"Filtered to {len(keep & set(ds.classes))} / {len(keep)} requested classes.")

    # ImageFolder gives alphabetic 0..K-1 indices over folder names; the pretrained backbone
    # outputs the original 0..N indices (1000 for ImageNet, 365 for Places365). Map
    # local→global via the dataset-specific CSV so the pred==gt comparison happens in the
    # same index space.
    mapping_path = args.mapping_csv
    simplified_to_global: dict[str, int] = {}
    with mapping_path.open() as f:
        for row in csv.DictReader(f):
            simplified_to_global[row["simplified"]] = int(row["idx"])
    missing = [c for c in ds.classes if c not in simplified_to_global]
    if missing:
        raise KeyError(f"{len(missing)} folder names not in {mapping_path.name}: {missing[:5]}")
    local_to_global = torch.tensor(
        [simplified_to_global[ds.classes[i]] for i in range(len(ds.classes))],
        dtype=torch.long,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.startswith("cuda")))

    correct, total = 0, 0
    written = 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(tqdm(loader)):
            x = x.to(device, non_blocking=True)
            preds = model(x).argmax(dim=-1).cpu()
            gt_global = local_to_global[y]
            for k, (pred, gt, gt_g) in enumerate(zip(preds, y, gt_global)):
                total += 1
                if int(pred) != int(gt_g):
                    continue
                correct += 1
                global_idx = i * args.batch_size + k
                src_path = Path(ds.samples[global_idx][0])
                cls_name = ds.classes[int(gt)]
                dst_dir = out_root / cls_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src_path.name
                if dst.exists() or dst.is_symlink():
                    continue
                if args.copy:
                    import shutil
                    shutil.copy2(src_path, dst)
                else:
                    try:
                        os.symlink(src_path.resolve(), dst)
                    except OSError:
                        # Fallback: copy if symlinks are disallowed (e.g. some Windows).
                        import shutil
                        shutil.copy2(src_path, dst)
                written += 1

    print(f"backbone={args.backbone}  acc={correct/total:.4f} ({correct}/{total})  wrote={written}")


if __name__ == "__main__":
    main()
