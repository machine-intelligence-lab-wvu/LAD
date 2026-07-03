"""ImageNet folder dataset, NPZ collation, and manifest building.

Consolidates the data-loading logic that lived in cells 14, 16, 18, 22, 23, 28, 30 of
``Imagenet_training_with_pipeline.ipynb``.

NPZ schema (per-image, written by :mod:`lad.clip_concepts.save_P_npz`):

    P:                 [r, h, w]     float32 — CLIP red-circle similarity tensor
    importance_scores: [r]           float32 — Otsu-mean magnitude per concept
    otsu_masks:        [r, h, w]     bool    — Otsu binarization of each P-slice
    concepts:          [r]           object  — concept text strings
    centers, grid_size, radius, image_hw, image_path  (metadata)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_transform(size: int = 224) -> transforms.Compose:
    """Standard ImageNet eval transform (resize → center-crop → tensor → normalize).

    ``size`` controls the square crop fed to the encoder; 448 doubles the encoder's spatial
    grid (7×7 → 14×14) for higher-resolution concept maps.
    """
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def list_images_by_class(images_root: Path) -> Tuple[List[str], dict[str, int], List[Tuple[Path, int]]]:
    """Walk ``images_root`` in ImageFolder layout. Returns ``(classes, class_to_idx, items)``."""
    images_root = Path(images_root)
    classes = sorted(d.name for d in images_root.iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items: List[Tuple[Path, int]] = []
    for c in classes:
        for p in (images_root / c).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((p, class_to_idx[c]))
    return classes, class_to_idx, items


def npz_suffix_from_variant(npz_variant: str) -> str:
    """Map ``NPZ_VARIANT`` strings to per-image NPZ filename suffixes."""
    if "r16_14x14" in npz_variant:
        return "__P_14x14_r16.npz"
    if "r32_7x7" in npz_variant:
        return "__P_7x7_r32.npz"
    raise ValueError(f"Unknown NPZ_VARIANT pattern: {npz_variant!r}")


def build_paired_items(
    images_root: Path,
    npz_root: Path,
    npz_suffix: str,
    *,
    verify_npz_key: Optional[str] = "P",
    classes: Optional[Iterable[str]] = None,
) -> Tuple[List[str], dict[str, int], List[Tuple[Path, int, Path, str]], dict[str, int]]:
    """Pair ``(image, npz)`` per image across an ImageFolder-style layout.

    Returns ``(classes, class_to_idx, paired_items, report)`` where each item is
    ``(img_path, label_idx, npz_path, id_str)``. Items missing an NPZ are silently dropped
    and counted in the report.
    """
    images_root, npz_root = Path(images_root), Path(npz_root)
    all_classes = sorted(d.name for d in images_root.iterdir() if d.is_dir())
    if classes is not None:
        keep = set(classes)
        all_classes = [c for c in all_classes if c in keep]
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    paired: List[Tuple[Path, int, Path, str]] = []
    report = {"num_classes": len(all_classes), "images_seen": 0, "paired": 0,
              "missing_npz": 0, "bad_npz": 0, "missing_key": 0}

    for c in all_classes:
        for img_path in (images_root / c).rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                continue
            report["images_seen"] += 1
            npz_path = npz_root / c / f"{img_path.stem}{npz_suffix}"
            if not npz_path.exists():
                report["missing_npz"] += 1
                continue
            if verify_npz_key is not None:
                try:
                    with np.load(npz_path, allow_pickle=True) as d:
                        if verify_npz_key not in d.files:
                            report["missing_key"] += 1
                            continue
                except Exception:
                    report["bad_npz"] += 1
                    continue
            id_str = f"{c}/{img_path.stem}"
            paired.append((img_path, class_to_idx[c], npz_path, id_str))
            report["paired"] += 1
    return all_classes, class_to_idx, paired, report


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _np_to_torch_safe(arr):
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.dtype == np.object_:
        return arr.item() if arr.shape == () else arr
    return torch.from_numpy(arr)


class ImagenetImageNpzDataset(torch.utils.data.Dataset):
    """Returns ``(image_tensor, label_int, npz_dict, id_str)`` per index.

    The NPZ dict has the keys listed in the module docstring, with numeric arrays converted
    to ``torch.Tensor`` and object arrays kept as Python objects.
    """

    def __init__(self, paired_items, transform=None):
        self.items = paired_items
        self.transform = transform or imagenet_transform()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        img_path, y, npz_path, id_str = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        with np.load(npz_path, allow_pickle=True) as data:
            npz_dict = {k: _np_to_torch_safe(data[k]) for k in data.files}
        return img, int(y), npz_dict, id_str


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_concept_npz(batch, *, p_key: str = "P", scores_key: str = "importance_scores",
                        mask_key: str = "otsu_masks"):
    """Collate function for :class:`ImagenetImageNpzDataset`.

    Stacks the numeric NPZ tensors and keeps the text/metadata fields as Python lists.

    Returns
    -------
    imgs : ``[B, 3, 224, 224]`` float
    ys : ``[B]`` long
    P : ``[B, r, h, w]`` (typically float16)
    scores : ``[B, r]`` float32
    masks : ``[B, r, h, w]`` bool
    concepts : ``list[length B]`` of per-image concept strings
    meta : ``dict`` of metadata lists (centers, grid_size, radius, image_hw, image_path)
    ids : ``list[str]`` per-image id strings
    """
    imgs, ys, npzs, ids = zip(*batch)
    imgs = torch.stack(imgs, 0)
    ys = torch.tensor(ys, dtype=torch.long)
    P = torch.stack([d[p_key] for d in npzs], dim=0)
    scores = torch.stack([d[scores_key] for d in npzs], dim=0)
    masks = torch.stack([d[mask_key] for d in npzs], dim=0)
    concepts = [d.get("concepts") for d in npzs]
    meta = {k: [d.get(k) for d in npzs] for k in ("centers", "grid_size", "radius", "image_hw", "image_path")}
    return imgs, ys, P, scores, masks, concepts, meta, list(ids)


# ---------------------------------------------------------------------------
# K-fold split (per-class) — keeps the train/val ratios from the paper
# ---------------------------------------------------------------------------


def per_class_kfold(items: List, n_folds: int, fold: int, val_ratio: float = 0.3, seed: int = 42):
    """Deterministic per-class KFold-ish split.

    Items are shuffled with a seed derived from ``seed + fold``, then the first
    ``val_ratio`` fraction goes to validation and the rest to training. This matches the
    notebook's ``rng = np.random.default_rng(42 + fold)`` pattern.
    """
    n = len(items)
    if n == 0:
        return [], []
    rng = np.random.default_rng(seed + fold)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_val = max(1, int(val_ratio * n))
    return idxs[n_val:].tolist(), idxs[:n_val].tolist()


__all__ = [
    "IMG_EXTS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "imagenet_transform",
    "list_images_by_class",
    "npz_suffix_from_variant",
    "build_paired_items",
    "ImagenetImageNpzDataset",
    "collate_concept_npz",
    "per_class_kfold",
]
