"""Single source of truth for filesystem paths.

Every script and module imports the constants below; nothing hard-codes an
absolute path. Override any of them at runtime via environment variables (see
``.env.example``) or by exporting them in your shell.

The defaults place all data and outputs under the repository root, so a fresh
clone works out of the box::

    <repo>/data/       downloaded images, filtered pools, CLIP NPZ caches
    <repo>/outputs/    trained concept bases (W), result CSVs, logs
    <repo>/models/     trained classifier checkpoints (Places365 / retinal)
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val else default


ROOT = _env_path("LAD_ROOT", Path(__file__).resolve().parent)

# --- data ---------------------------------------------------------------------
DATA_ROOT = _env_path("LAD_DATA_ROOT", ROOT / "data")
IMAGENET_ROOT = _env_path("LAD_IMAGENET_ROOT", DATA_ROOT / "imagenet")
FILTERED_ROOT = _env_path("LAD_FILTERED_ROOT", DATA_ROOT / "imagenet_filtered")
NPZ_ROOT = _env_path("LAD_NPZ_ROOT", DATA_ROOT / "imagenet_npz")

# --- concept vocabularies (versioned in the repo) -----------------------------
CONCEPTS_DIR = _env_path("LAD_CONCEPTS_DIR", ROOT / "concept_outputs" / "per_class_txt")
CLASSES_FILE = _env_path("LAD_CLASSES_FILE", ROOT / "concept_outputs" / "classes_500.txt")

# --- retinal (ODIR-5K) clinical track -----------------------------------------
# RETINAL_ROOT holds the organized class-folder pool + manifest produced by
# scripts/retinal/00_prepare_odir.py. The raw ODIR source archive is passed at
# runtime (--odir-src) and is never hard-coded here.
RETINAL_ROOT = _env_path("LAD_RETINAL_ROOT", DATA_ROOT / "odir")
RETINAL_FILTERED_ROOT = _env_path("LAD_RETINAL_FILTERED_ROOT", RETINAL_ROOT / "filtered")
RETINAL_NPZ_ROOT = _env_path("LAD_RETINAL_NPZ_ROOT", RETINAL_ROOT / "npz")
RETINAL_CONCEPTS_DIR = _env_path("LAD_RETINAL_CONCEPTS_DIR", ROOT / "concept_outputs" / "retinal")

# --- trained classifier checkpoints -------------------------------------------
MODELS_DIR = _env_path("LAD_MODELS_DIR", ROOT / "models")

# --- outputs ------------------------------------------------------------------
OUTPUT_ROOT = _env_path("LAD_OUTPUT_ROOT", ROOT / "outputs")
GRID_OUTPUT_ROOT = _env_path("LAD_GRID_OUTPUT_ROOT", OUTPUT_ROOT / "grid")
LOGS_DIR = _env_path("LAD_LOGS_DIR", OUTPUT_ROOT / "logs")


def w_path(backbone: str, npz_variant: str, class_name: str, fold: int) -> Path:
    """Canonical path for a saved concept basis W."""
    return GRID_OUTPUT_ROOT / backbone / npz_variant / f"fold_{fold}" / class_name / "W_final.pt"


def npz_path(backbone: str, npz_variant: str, class_name: str, image_id: str) -> Path:
    """Canonical path for a per-image CLIP-S NPZ."""
    return NPZ_ROOT / backbone / npz_variant / "correct" / class_name / f"{image_id}.npz"


def filtered_class_dir(backbone: str, class_name: str) -> Path:
    return FILTERED_ROOT / backbone / "correct" / class_name
