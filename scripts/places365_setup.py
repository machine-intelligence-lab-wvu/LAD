"""One-time Places365 dataset setup for the LAD pipeline.

Steps:

1. Download the Places365 dataset via ``kagglehub`` (``benjaminkz/places365``).
2. Detect ``val/`` layout (per-class dirs vs flat JPEGs) and reorganize into the
   per-class ImageFolder layout LAD expects, using ``val.txt`` for labels.
3. Place a copy of ``categories_places365.txt`` (label-index → name mapping) at
   ``concept_outputs/places365_categories.txt`` and emit a simplified class list at
   ``concept_outputs/classes_places365.txt`` (one normalized class name per line) for
   feeding into ``03_generate_concepts.py`` and the rest of the pipeline.
4. Symlink/move the per-class val/ tree to ``$LAD_FILTERED_ROOT/raw_places365/`` so the
   existing ``02_filter_correct.py`` (which expects an ImageFolder) can run on it.

Notes:
- Places365 categories use the slash notation ``/a/abbey``, ``/m/music_studio`` for the
  hierarchical layout under MIT's filesystem. We strip the leading letter prefix and
  flatten ``/`` → ``__`` in the filesystem (``abbey``, ``music_studio``) to mirror the
  ImageNet "simplified" names format used in ``concept_outputs/imagenet1k_classes.csv``.
- The pretrained ResNet-50 Places365 backbone downloads on demand via
  ``lad.backbones.load_backbone('resnet50_places365')``; nothing to fetch here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
import paths  # type: ignore  # noqa: E402

# Mirror of the canonical CSAILVision file (also reproducible from the train/val.txt
# manifests if needed; the URL below is the standard reference).
_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle-dataset", default="benjaminkz/places365",
                   help="Kaggle dataset slug for kagglehub.")
    p.add_argument("--out-images-root", type=Path,
                   default=paths.DATA_ROOT / "places365_raw",
                   help="Where the per-class val/ tree will live (input to step 02).")
    p.add_argument("--out-classes-file", type=Path,
                   default=paths.ROOT / "concept_outputs" / "classes_places365.txt",
                   help="Will be filled with one simplified class name per line.")
    p.add_argument("--out-categories-file", type=Path,
                   default=paths.ROOT / "concept_outputs" / "places365_categories.txt",
                   help="Canonical CSAILVision categories file (path label-idx).")
    p.add_argument("--out-mapping-csv", type=Path,
                   default=paths.ROOT / "concept_outputs" / "places365_classes.csv",
                   help="CSV with idx,raw,simplified columns matching imagenet1k_classes.csv "
                        "format. Pass this via --mapping-csv to 02_filter_correct.py.")
    p.add_argument("--limit-classes", type=int, default=None,
                   help="If set, only stage the first N alphabetical classes (for a small "
                        "rebuttal subset; full set has 365).")
    return p.parse_args()


def simplified(raw: str) -> str:
    """Convert ``/a/airfield`` or ``/m/music_studio/indoor`` to ``airfield`` / ``music_studio_indoor``."""
    raw = raw.lstrip("/")
    parts = raw.split("/")
    # Drop the first letter-prefix folder ('/a', '/b', etc.)
    if len(parts) > 1 and len(parts[0]) == 1:
        parts = parts[1:]
    return "_".join(parts)


def fetch_categories(out_path: Path) -> list[tuple[int, str, str]]:
    """Return list of (idx, raw_name, simplified_name); also writes the raw file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        print(f"Downloading categories file → {out_path}")
        urllib.request.urlretrieve(_CATEGORIES_URL, out_path)
    rows: list[tuple[int, str, str]] = []
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw_name, idx_str = line.rsplit(" ", 1)
        rows.append((int(idx_str), raw_name, simplified(raw_name)))
    rows.sort(key=lambda t: t[0])
    return rows


def kagglehub_download(slug: str) -> Path:
    """Download via kagglehub; returns the local path. User must have kaggle credentials."""
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "kagglehub not installed. Run: pip install kagglehub  "
            "(and ensure ~/.kaggle/kaggle.json is in place with chmod 600)."
        ) from e
    print(f"Downloading via kagglehub: {slug} (this may take a while; ~25 GB total) ...")
    path = Path(kagglehub.dataset_download(slug))
    print(f"  downloaded to: {path}")
    return path


def detect_val_layout(val_dir: Path) -> str:
    """Return 'per_class' if val/ contains class subdirs, else 'flat'."""
    subdirs = [p for p in val_dir.iterdir() if p.is_dir()]
    files = [p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(subdirs) > 50 and len(files) == 0:
        return "per_class"
    if len(files) > 100 and len(subdirs) == 0:
        return "flat"
    raise RuntimeError(f"Unexpected val/ layout at {val_dir}: {len(subdirs)} dirs, {len(files)} files")


def reorganize_flat_val(
    val_dir: Path, val_txt: Path, out_root: Path,
    idx_to_simplified: dict[int, str],
    limit_classes: int | None = None,
) -> int:
    """Move flat val/<filename>.jpg to out_root/<class_name>/<filename>.jpg using val.txt mapping."""
    out_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    keep_idxs: set[int] | None = None
    if limit_classes is not None:
        keep_idxs = set(sorted(idx_to_simplified.keys())[:limit_classes])
        print(f"  (limiting to first {limit_classes} class indices: {sorted(keep_idxs)[:5]}...)")

    for line in val_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # val.txt format: "<filename> <label_idx>" (filename may include sub-paths)
        rel, idx_str = line.rsplit(" ", 1)
        idx = int(idx_str)
        if keep_idxs is not None and idx not in keep_idxs:
            continue
        cls = idx_to_simplified.get(idx)
        if cls is None:
            continue
        src = val_dir / rel.lstrip("/")
        if not src.exists():
            # Some Kaggle dumps strip the directory prefix; try just the basename
            src = val_dir / Path(rel).name
            if not src.exists():
                continue
        dst_dir = out_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            moved += 1
    return moved


def reorganize_per_class_val(
    val_dir: Path, out_root: Path,
    idx_to_simplified: dict[int, str], raw_to_simplified: dict[str, str],
    limit_classes: int | None = None,
) -> int:
    """Already per-class — copy/symlink each class folder under out_root with simplified name."""
    out_root.mkdir(parents=True, exist_ok=True)
    keep_simplified: set[str] | None = None
    if limit_classes is not None:
        keep_simplified = {idx_to_simplified[i] for i in sorted(idx_to_simplified.keys())[:limit_classes]}
    moved = 0
    for src_dir in val_dir.iterdir():
        if not src_dir.is_dir():
            continue
        # The per-class folder name might be the raw '/a/abbey'-style or already 'abbey'.
        raw_key = "/" + src_dir.name if not src_dir.name.startswith("/") else src_dir.name
        cls = raw_to_simplified.get(raw_key, simplified(src_dir.name))
        if keep_simplified is not None and cls not in keep_simplified:
            continue
        dst_dir = out_root / cls
        if not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir)
            moved += sum(1 for _ in dst_dir.iterdir())
    return moved


def main():
    args = parse_args()

    # 1. Download
    root = kagglehub_download(args.kaggle_dataset)
    val_dir = root / "val"
    val_txt = root / "val.txt"
    if not val_dir.is_dir():
        raise SystemExit(f"Expected {val_dir} — Kaggle dataset layout differs from anticipated")

    # 2. Categories
    rows = fetch_categories(args.out_categories_file)
    idx_to_simplified = {idx: simp for (idx, _raw, simp) in rows}
    raw_to_simplified = {raw: simp for (_idx, raw, simp) in rows}
    print(f"Loaded {len(rows)} Places365 categories.")

    # 3. Reorganize val/ into per-class layout
    layout = detect_val_layout(val_dir)
    print(f"val/ layout detected as: {layout}")
    if layout == "flat":
        if not val_txt.exists():
            raise SystemExit(f"Need {val_txt} for flat→per-class reorg but it's missing.")
        n = reorganize_flat_val(val_dir, val_txt, args.out_images_root,
                                idx_to_simplified, limit_classes=args.limit_classes)
    else:
        n = reorganize_per_class_val(val_dir, args.out_images_root,
                                     idx_to_simplified, raw_to_simplified,
                                     limit_classes=args.limit_classes)
    print(f"Staged {n} images into {args.out_images_root}")

    # 4. Class manifest (simplified names, one per line)
    classes = sorted({p.name for p in args.out_images_root.iterdir() if p.is_dir()})
    args.out_classes_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_classes_file.write_text("\n".join(classes) + "\n")
    print(f"Wrote {len(classes)} classes to {args.out_classes_file}")
    print(f"Sample: {classes[:5]}")

    # 5. CSV mapping (idx,raw,simplified) — same shape as imagenet1k_classes.csv,
    #    consumed by 02_filter_correct.py via --mapping-csv.
    import csv as _csv
    args.out_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_mapping_csv.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["idx", "raw", "simplified"])
        for idx, raw, simp in rows:
            w.writerow([idx, raw, simp])
    print(f"Wrote mapping CSV to {args.out_mapping_csv}")

    print("\nNext steps (cluster):")
    print(f"  Step 2: python scripts/02_filter_correct.py --backbone resnet50_places365 \\")
    print(f"          --imagenet-root {args.out_images_root} --no-split-subdir \\")
    print(f"          --mapping-csv {args.out_mapping_csv} \\")
    print(f"          --classes-file {args.out_classes_file}")
    print(f"  Step 3: python scripts/03_generate_concepts.py \\")
    print(f"          --classes-file {args.out_classes_file}")
    print(f"  Step 4 onwards: standard pipeline with --backbone resnet50_places365 \\")
    print(f"          --classes-file {args.out_classes_file}")


if __name__ == "__main__":
    main()
