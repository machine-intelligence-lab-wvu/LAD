# LAD — Language-Anchored Decomposition

**Naming the Concepts Classifiers Rely On: Language-Anchored Decomposition for Faithful Explanation**

LAD is a *post-hoc* concept-discovery framework whose concepts are simultaneously **named**, **faithful**, and obtained **without modifying the model**. For each class, a language model proposes a concept vocabulary and CLIP-based similarity maps localize each concept across image regions. LAD then **inverts** non-negative matrix factorization: it *fixes* the language-grounded similarity maps as the coefficient matrix `S` and learns *only* the concept basis `W` that reconstructs a frozen encoder's activations. Naming therefore becomes a structural constraint, and the model's own feature geometry — not CLIP — decides which concepts survive.

The optimization is

```
min_{W ≥ 0}  ½ · ‖ Ā − S Wᵀ ‖²_F        (learn the basis W by projected gradient descent)
```

where `Ā` are the unfolded encoder activations (one row per spatial location). At inference, per-image concept coefficients `Ŝ` are recovered in closed form (non-negative least squares) with optional PGD refinement, then reshaped into named concept heatmaps.

> Removing the language anchor (learning `S` too, i.e. plain unsupervised NMF) leaves **accuracy unchanged** but **collapses deletion faithfulness** — the anchor determines *which* concept directions are discovered, it is not cosmetic.

## Results

Primary comparison (from the paper). All methods preserve accuracy, so faithfulness is read from concept **insertion** (C-Ins ↑, are the concepts *sufficient*) and **deletion** (C-Del ↑, are they *necessary*).

| Dataset (backbone, #classes) | Method | Acc ↑ | C-Ins ↑ | C-Del ↑ |
|---|---|---:|---:|---:|
| **ImageNet** (ResNet34, 500) | **LAD** | **1.000** | **0.973** | 0.902 |
|  | FACE | 0.998 | 0.952 | **0.939** |
|  | CRAFT | 0.976 | 0.918 | 0.808 |
|  | ICE | 0.997 | 0.907 | 0.545 |
| **Places365** (ResNet50, 364) | **LAD** | **1.000** | **0.972** | 0.880 |
|  | FACE | 0.994 | 0.925 | **0.919** |
|  | CRAFT | 0.947 | 0.868 | 0.828 |
|  | ICE | 0.995 | 0.887 | 0.631 |
| **Retinal ODIR-5K** (DenseNet-121, BiomedCLIP) | **LAD** | **1.000** | 0.980 | **0.954** |
|  | FACE | 0.990 | 0.980 | 0.863 |
|  | CRAFT | 1.000 | 0.980 | 0.850 |
|  | ICE | 0.990 | 0.980 | 0.455 |

LAD leads on insertion everywhere and **uniquely names its concepts**; on the focal-evidence clinical domain it also leads deletion by a wide margin.

## Installation

```bash
git clone https://github.com/<your-org>/lad.git && cd lad
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .[dev]

cp .env.example .env                 # then fill in OPENAI_API_KEY / HF_TOKEN (see below)
python -m pytest tests/ -v           # 10 fast CPU-only unit tests
```

A CUDA GPU is required to build CLIP maps and train at scale; the unit tests and the
one-class smoke check run on CPU.

**Keys.** `OPENAI_API_KEY` is only needed to *regenerate* concept vocabularies (Step 3);
the vocabularies used in the paper are already committed under `concept_outputs/`, so you
can reproduce everything without it. `HF_TOKEN` is needed to download ImageNet / Places365
(Step 1) from HuggingFace.

**Paths.** Everything resolves through `paths.py`, driven by `LAD_*` environment variables
(see `.env.example`). Defaults live under the repo root (`data/`, `outputs/`, `models/`). Point
`LAD_DATA_ROOT` / `LAD_OUTPUT_ROOT` at a large disk if you don't want them inside the repo.

## Repository layout

```
src/lad/
  decomposition.py   the math: unfold ops, NNDSVD/NNLS init, PGD trainer (train_W_pgd),
                     inference (solve_S_hat), and the FACE baseline (KL_NMF)
  clip_concepts.py   LLM concept-vocabulary generation + red-circle CLIP S-matrix builder
  backbones.py       encoder g(x)→[B,C,h,w] + head h_2d for resnet*/mobilenet/convnext/
                     vit_b_16/resnet50_places365; custom-head classifiers for the clinical track
  data.py            image/NPZ datasets, paired manifests, per-class K-fold splitter
  metrics.py         accuracy, Sobol' importance, Gini, insertion/deletion AUC, sparsity
  baselines/craft/   vendored CRAFT (Fel et al., CVPR 2023)      — see ATTRIBUTION.md
  baselines/ice/     vendored InvertibleCE / ICE (Zhang et al.)  — see ATTRIBUTION.md
scripts/             the numbered pipeline (Step 1–9) + ablations + figure scripts
  retinal/           the clinical-domain (ODIR-5K) pipeline
concept_outputs/     committed concept vocabularies + class lists (no need to re-run the LLM)
  per_class_txt/            ImageNet, r=25 (500 classes)
  per_class_txt_c{10,50,100}/  ImageNet, r∈{10,50,100} for the concept-budget ablation
  per_class_txt_places365/ Places365 (364 classes)
  retinal/                 ODIR-5K disease vocabularies
notebooks/
  01_sanity_check.ipynb            end-to-end smoke on one class
  03_retinal_concept_explorer.ipynb  interactive retinal concept maps
paths.py             single source of truth for all filesystem paths
```

## Quick check (no dataset download)

Runs the core pipeline (concepts → CLIP maps → train → eval) end-to-end on a single class and writes a top-k concept heatmap:

```bash
python scripts/dry_run.py --class african_elephant --limit 15
```

Edit `ARCHIVE_ROOT` inside the script to point at a local image folder, or pass
`--hf-download` to stream a few images from HuggingFace.

---

## Reproducing the paper — ImageNet (main result)

The pipeline is a chain of numbered scripts. Each `(class, fold)` is independent, so Step 4/5
parallelize trivially across a machine or cluster. Defaults reproduce the paper: **ResNet34
backbone, 500 classes, rank r=25, K=3 folds (fold 0 reported), CLIP ViT-B/16 red-circle maps
on a 14×14 grid (radius 16), 150 PGD epochs.**

### Step 1 — Get the images

```bash
python scripts/01_download_imagenet.py            # 500-class subset, both splits
```

Downloads the gated HuggingFace `imagenet-1k` into `data/imagenet/{train,validation}/<class>/`
using the canonical simplified class names. Idempotent (resumes if interrupted). Already have
ImageNet locally? Skip this and point `LAD_IMAGENET_ROOT` at it.

### Step 2 — Keep correctly-classified images

```bash
python scripts/02_filter_correct.py --backbone resnet34 --split train
```

Forward-passes the frozen backbone and symlinks the images it classifies correctly into
`data/imagenet_filtered/resnet34/correct/<class>/`. Concept discovery runs per class on this
pool.

### Step 3 — Concept vocabularies *(optional — already committed)*

The paper's vocabularies live in `concept_outputs/per_class_txt/`. Only re-run this to
regenerate them (needs `OPENAI_API_KEY`):

```bash
python scripts/03_generate_concepts.py --num-concepts 25
```

### Step 4 — Build the CLIP concept maps (the fixed `S`)

```bash
python scripts/04_build_clip_npz.py --backbone resnet34 --grid 14x14 --radius 16
```

For every filtered image, overlays a red circle at each grid cell and records CLIP
image–text similarity to each concept, saving a per-image NPZ under
`data/imagenet_npz/resnet34/ViT-B-16_r16_14x14/correct/<class>/`. Idempotent. This is the
most expensive step (one CLIP forward per grid cell per concept per image).

### Step 5 — Learn the concept basis `W` (LAD)

One job per `(class, fold)`. Single class:

```bash
python scripts/05_train_lad.py --backbone resnet34 --class golden_retriever \
    --fold 0 --rank 25 --epochs 150
```

All 500 classes, fold 0 (bash — one line per class in `classes_500.txt`):

```bash
while read c; do
  [ -z "$c" ] && continue
  python scripts/05_train_lad.py --backbone resnet34 --class "$c" --fold 0 --rank 25 --epochs 150
done < concept_outputs/classes_500.txt
```

PowerShell:

```powershell
Get-Content concept_outputs\classes_500.txt | Where-Object { $_ } | ForEach-Object {
  python scripts\05_train_lad.py --backbone resnet34 --class $_ --fold 0 --rank 25 --epochs 150
}
```

Each run writes `outputs/grid/resnet34/ViT-B-16_r16_14x14/fold_0/<class>/W_final.pt`.

### Step 6 — Evaluate

```bash
python scripts/06_eval_metrics.py --include-folds 0 \
    --output outputs/all_classes_results.csv
```

Walks every trained `W_final.pt` and writes one CSV row per class with Acc, C-Ins AUC,
C-Del AOPC, Gini, and sparsity. The column means are the ImageNet row of the results table.

---

## Baselines (FACE, CRAFT, ICE)

Same backbone, preprocessing, splits, concept count, and CLIP maps as LAD — only the
factorization differs. Train and evaluate per `(class, fold)`:

```bash
# FACE (joint U+W with a KL logit-alignment term)
python scripts/07_train_face.py  --backbone resnet34 --class golden_retriever --fold 0 --rank 25
python scripts/09_eval_baselines.py --mode face \
    --grid-root outputs/grid_face  --output outputs/results_face.csv --include-folds 0

# CRAFT (sklearn NMF, joint U+W)
python scripts/08_train_craft.py --backbone resnet34 --class golden_retriever --fold 0 --rank 25
python scripts/09_eval_baselines.py --mode craft \
    --grid-root outputs/grid_craft --output outputs/results_craft.csv --include-folds 0
```

Loop over `concept_outputs/classes_500.txt` exactly as in Step 5 to cover all classes.
(ICE is evaluated through the vendored `src/lad/baselines/ice/`; the retinal baseline
runner below drives all four methods including ICE in one command.)

---

## Ablations

**Language anchor (paper Table 3).** Replace the fixed `S` with a learned one — i.e. plain
unsupervised NMF on the same activations — then evaluate the same way. Accuracy is unchanged
but C-Del collapses (0.902 → 0.507):

```bash
python scripts/train_lad_ablation_no_lang.py --backbone resnet34 --class golden_retriever --fold 0 --rank 25
```

`train_lad_ablation_no_unfold.py` is the companion spatial-unfolding ablation.

**Concept budget r (paper Table 4).** Re-run Steps 4–6 with `--rank {10,50,100}`, using the
matching committed vocabularies `concept_outputs/per_class_txt_c{10,50,100}/` via
`--concepts-dir`. Faithfulness improves monotonically with r; accuracy stays saturated.

---

## Places365 (scene benchmark)

Mirrors the ImageNet pipeline with the Places365-pretrained ResNet50 (auto-downloaded by
`backbones.load_backbone("resnet50_places365")`) and the Places class list + vocabularies.

```bash
# 1. Materialize Places365 into class folders (writes the class list + idx→name mapping csv)
python scripts/places365_setup.py

# 2–4. Filter / build CLIP maps against the Places backbone and class list
python scripts/02_filter_correct.py --backbone resnet50_places365 \
    --classes-file concept_outputs/classes_places365.txt \
    --mapping-csv concept_outputs/places365_classes.csv
python scripts/04_build_clip_npz.py --backbone resnet50_places365 --grid 14x14 --radius 16 \
    --classes-file concept_outputs/classes_places365.txt \
    --concepts-dir concept_outputs/per_class_txt_places365

# 5–6. Train + evaluate (loop over classes_places365.txt as in Step 5)
python scripts/05_train_lad.py  --backbone resnet50_places365 --class <class> --fold 0 --rank 25 --epochs 150
python scripts/06_eval_metrics.py --include-folds 0 --output outputs/results_places365.csv
```

---

## Retinal ODIR-5K (clinical domain)

A DenseNet-121 fine-tuned on ODIR-5K color-fundus photographs, with concepts localized by
**BiomedCLIP** at 448px. The eval scripts learn `W` and score in one shot, so no separate
NPZ cache is required.

```bash
# 0. Organize the ODIR-5K source archive into class folders (+ manifest)
python scripts/retinal/00_prepare_odir.py --odir-src /path/to/ODIR-5K

# 1. Fine-tune the classifier LAD will explain
python scripts/retinal/01_train_classifier.py --backbone densenet121

# 2. Keep correctly-classified fundus images
python scripts/retinal/02_filter_correct.py --backbone densenet121

# 3. Concept vocabularies (optional — committed in concept_outputs/retinal/)
python scripts/retinal/03_generate_concepts.py --clip biomedclip

# 6/7. Evaluate LAD, and LAD vs FACE/CRAFT/ICE
python scripts/retinal/06_eval_retinal.py   --backbone densenet121 --clip biomedclip
python scripts/retinal/07_eval_baselines.py --methods lad face craft ice
```

`notebooks/03_retinal_concept_explorer.ipynb` renders the named per-disease concept maps
(e.g. *drusen-like structures*, *vascular dilation*) interactively.

---

## Figures

The named-concept rollouts and qualitative comparisons in the paper are produced by
`scripts/visualize_concepts_batch.py` and `scripts/figures/render_*.py` (rollout,
concept-vs-factor, gallery) from trained `W` files.

## Citation

```bibtex
@article{akash2026lad,
  title   = {Naming the Concepts Classifiers Rely On: Language-Anchored Decomposition for Faithful Explanation},
  author  = {Akash, Ahsan Habib and Bhusal, Dipkamal and Jones, Stacey and
             Adjeroh, Donald A. and Bhattarai, Binod and Gyawali, Prashnna Kumar},
  year    = {2026}
}
```

## License & acknowledgements

Code released under the MIT License (`LICENSE`). The `src/lad/baselines/craft/` and
`src/lad/baselines/ice/` directories vendor prior work (CRAFT — Fel et al., CVPR 2023;
InvertibleCE — Zhang et al., AAAI 2021) under their original licenses; see the
`ATTRIBUTION.md` in each.
