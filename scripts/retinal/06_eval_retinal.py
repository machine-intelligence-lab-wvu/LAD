"""Step-6 eval for the retinal LAD maps: Acc, C-Ins, C-Del per disease class.

Mirrors scripts/06_eval_metrics.py (legacy image-level insertion/deletion) but for the
retinal classifier + current backbones API. Per class:
  1. split the correctly-classified images into train/val (per_class_kfold, fold 0);
  2. learn the concept basis W on train (same probing as the figures: input 448, 28x28
     circle grid pooled to the 14x14 encoder, radius 8, BiomedCLIP);
  3. on val: recover U=NNLS(A,W), then report
       Acc    = classification accuracy after reconstructing activations from concepts,
       C-Ins  = insertion AUC (add concepts most-important-first),
       C-Del  = deletion AOPC (remove concepts most-important-first),
     plus reconstruction-free Acc, Gini, sparsity for context.

Usage: python scripts/retinal/06_eval_retinal.py --ckpt-suffix 448 --input-size 448
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402
from lad.backbones import load_classifier_checkpoint, make_g_and_h2d  # noqa: E402
from lad.clip_concepts import load_clip_model  # noqa: E402
from lad.data import imagenet_transform, per_class_kfold  # noqa: E402
from lad.decomposition import flatten_hw_rowmajor, nonneg_ls_init_U_hybrid  # noqa: E402
from lad.metrics import (  # noqa: E402
    calculate_gini,
    compute_deletion_score,
    compute_insertion_auc,
    compute_sparsity,
    concept_deletion,
    concept_insertion,
    estimate_importance,
)
from lad.retinal import probe_and_learn  # noqa: E402

DISEASES = ["diabetic_retinopathy", "glaucoma", "cataract", "amd"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="densenet121")
    p.add_argument("--ckpt-suffix", default="448")
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--clip", default="biomedclip")
    p.add_argument("--grid", default="28x28")
    p.add_argument("--radius", type=int, default=8)
    p.add_argument("--classes", nargs="+", default=DISEASES)
    p.add_argument("--max-train", type=int, default=70, help="Cap train images used to learn W.")
    p.add_argument("--max-val", type=int, default=40, help="Cap val images used to evaluate.")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "retinal" / "eval_retinal_448.csv")
    return p.parse_args()


def val_activations_and_U(val_imgs, g, W, input_size, device):
    tf = imagenet_transform(input_size)
    A_chunks = []
    hw = None
    with torch.inference_mode():
        for ip in val_imgs:
            x = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
            A_flat, _idx, hw = flatten_hw_rowmajor(g(x)[0])  # g -> (z, side); CNN side is None
            A_chunks.append(A_flat.cpu())
    A = torch.cat(A_chunks, 0).to(device).float()
    U = nonneg_ls_init_U_hybrid(A, W, pgd_iters=30)
    return A, U, hw


def main():
    args = parse_args()
    dev = args.device
    grid = tuple(int(x) for x in args.grid.split("x"))
    ckpt = paths.MODELS_DIR / f"best_{args.backbone}_odir{args.ckpt_suffix}.pt"

    clip_model, _, clip_tok, _ = load_clip_model(args.clip, device=dev)
    bundle = (clip_model, clip_tok)
    classifier = load_classifier_checkpoint(ckpt, device=dev)
    _model, backbone, classes = classifier
    g, h_2d = make_g_and_h2d(_model, backbone, device=dev)
    print(f"eval {ckpt.name}  input={args.input_size}  grid={grid} r{args.radius}  classes={args.classes}\n")

    rows = []
    for cls in args.classes:
        target = classes.index(cls)
        img_dir = paths.RETINAL_FILTERED_ROOT / backbone / "correct" / cls
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        tr_idx, va_idx = per_class_kfold(imgs, n_folds=3, fold=0, val_ratio=0.3, seed=42)
        train = [imgs[i] for i in tr_idx][: args.max_train]
        val = [imgs[i] for i in va_idx][: args.max_val]
        t0 = time.perf_counter()

        # learn W on train (same probing as the figures)
        r = probe_and_learn(cls, args.clip, train, checkpoint=ckpt,
                            concepts_dir=paths.RETINAL_CONCEPTS_DIR, device=dev, grid=grid,
                            radius=args.radius, epochs=args.epochs, classifier=classifier,
                            clip_bundle=bundle, input_size=args.input_size)
        W = r["W"].to(dev).float()           # [C, r]
        rank = W.shape[1]

        # recover U on val and score
        A, U, hw = val_activations_and_U(val, g, W, args.input_size, dev)
        B = len(val); h, w = hw; L = h * w

        with torch.inference_mode():
            A_hat = (U @ W.T).reshape(B, h, w, -1).permute(0, 3, 1, 2)
            A_orig = A.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            acc_recon = float((h_2d(A_hat).argmax(-1).cpu() == target).float().mean())
            acc_orig = float((h_2d(A_orig).argmax(-1).cpu() == target).float().mean())

        U_per_image = U.reshape(B, L, rank).mean(dim=1)
        importance = estimate_importance(U_per_image.cpu().numpy(), W.T.cpu().numpy(),
                                         h_2d, h_2d, target, 64, rank, dev)
        gini = float(calculate_gini(importance))
        sparsity = float(compute_sparsity(U_per_image))

        U_spatial = U.reshape(B, h, w, rank).cpu()
        W_io = W.T.cpu()                      # [r, C]
        ins = concept_insertion(U_spatial, W_io, h_2d, target, importance)
        dele = concept_deletion(U_spatial, W_io, h_2d, target, importance)
        c_ins = float(compute_insertion_auc(ins))
        c_del = float(compute_deletion_score(dele))

        row = {"class": cls, "n_train": len(train), "n_val": B, "rank": rank,
               "acc_recon": round(acc_recon, 4), "acc_orig": round(acc_orig, 4),
               "c_ins_auc": round(c_ins, 4), "c_del_aopc": round(c_del, 4),
               "gini": round(gini, 4), "sparsity": round(sparsity, 4),
               "elapsed_s": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        print(f"{cls:<22} Acc={row['acc_recon']:.3f}  C-Ins={row['c_ins_auc']:.3f}  "
              f"C-Del={row['c_del_aopc']:.3f}  (acc_orig={row['acc_orig']:.3f}, "
              f"gini={row['gini']:.3f}, n_val={B}, {row['elapsed_s']:.0f}s)")

    # overall mean
    keys = ["acc_recon", "acc_orig", "c_ins_auc", "c_del_aopc", "gini", "sparsity"]
    mean = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
        wtr.writerow({"class": "MEAN", **mean})
    print("\n" + "=" * 60)
    print(f"{'MEAN':<22} Acc={mean['acc_recon']:.3f}  C-Ins={mean['c_ins_auc']:.3f}  "
          f"C-Del={mean['c_del_aopc']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
