"""Step 1: materialize ImageNet-1K (or the paper's 500-class subset) into a
class-folder layout that the rest of the pipeline consumes:

    <LAD_IMAGENET_ROOT>/train/<class>/<id>.JPEG
    <LAD_IMAGENET_ROOT>/validation/<class>/<id>.JPEG

Images are pulled from the gated HuggingFace `imagenet-1k` dataset, so you need a
HuggingFace token with access (set `HF_TOKEN`, or run `huggingface-cli login`).
Class folders are named with the canonical *simplified* names from
`concept_outputs/imagenet1k_classes.csv`, the same names used everywhere else.

By default only the classes listed in `concept_outputs/classes_500.txt` are
written (saves disk + time); pass `--all-classes` to download all 1000.

The download is idempotent — images already on disk are skipped, so re-running
resumes an interrupted download.

Already have ImageNet in the standard `train/<wnid>/` layout? Skip this script and
point `LAD_IMAGENET_ROOT` at a directory whose class folders use these simplified
names (or symlink them).

Examples
--------
    # paper subset (500 classes), both splits
    python scripts/01_download_imagenet.py

    # everything
    python scripts/01_download_imagenet.py --all-classes
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path
import paths  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagenet-root", type=Path, default=paths.IMAGENET_ROOT)
    p.add_argument("--mapping-csv", type=Path, default=paths.ROOT / "concept_outputs" / "imagenet1k_classes.csv")
    p.add_argument("--classes-file", type=Path, default=paths.CLASSES_FILE,
                   help="Restrict to these simplified class names (default: classes_500.txt).")
    p.add_argument("--all-classes", action="store_true", help="Download all 1000 classes.")
    p.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation"])
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--quality", type=int, default=95)
    args = p.parse_args()

    from datasets import load_dataset

    idx_to_name: dict[int, str] = {}
    with args.mapping_csv.open() as f:
        for row in csv.DictReader(f):
            idx_to_name[int(row["idx"])] = row["simplified"]
    assert len(idx_to_name) == 1000, f"expected 1000 classes, got {len(idx_to_name)}"

    keep_idx = None
    if not args.all_classes and args.classes_file.exists():
        keep_names = {ln.strip() for ln in args.classes_file.read_text().splitlines()
                      if ln.strip() and not ln.startswith("#")}
        keep_idx = {i for i, n in idx_to_name.items() if n in keep_names}
        print(f"restricting to {len(keep_idx)} of {len(keep_names)} classes from {args.classes_file}")

    out = args.imagenet_root
    ds = load_dataset("imagenet-1k", num_proc=args.num_proc)

    for split_name in args.splits:
        split = ds[split_name]
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        written = skipped = errors = 0
        print(f"=== {split_name}: {len(split)} examples ===", flush=True)
        for i, ex in enumerate(split):
            if keep_idx is not None and ex["label"] not in keep_idx:
                continue
            cls_dir = split_dir / idx_to_name[ex["label"]]
            cls_dir.mkdir(exist_ok=True)
            out_path = cls_dir / f"{split_name}_{i:08d}.JPEG"
            if out_path.exists():
                skipped += 1
                continue
            try:
                img = ex["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(out_path, "JPEG", quality=args.quality)
                written += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors <= 20:
                    print(f"  WARN: failed {out_path.name}: {exc!r}", file=sys.stderr, flush=True)
            if i % 50000 == 0:
                print(f"  {split_name}: {i}/{len(split)}  written={written} skipped={skipped} errors={errors}",
                      flush=True)
        print(f"=== {split_name} done: written={written} skipped={skipped} errors={errors} ===", flush=True)

    n_train = len(list((out / "train").glob("*"))) if (out / "train").exists() else 0
    print(f"done — {n_train} train class folders under {out}")


if __name__ == "__main__":
    main()
