"""Retinal baseline comparison: LAD vs FACE vs CRAFT vs ICE on Acc / C-Ins / C-Del.

For each disease class, learns a concept basis W=[C,r] with each method on a train split and
scores it on a held-out val split with **identical** metric code (so differences reflect the
basis, not the eval):
  * LAD  — CLIP-guided decomposition (probe_and_learn; the method behind the figures).
  * FACE  — KL-regularized NMF on image-patch activations (train_face_W).
  * CRAFT — plain NMF on image-patch activations (train_craft_W).
  * ICE   — NMF on the grid-cell activations directly (ChannelDecompositionReducer).

All scored on val via U=NNLS(A,W): Acc (recon classification), C-Ins (insertion AUC),
C-Del (deletion AOPC), plus Gini. densenet121 @448, BiomedCLIP, rank 25.

Usage: python scripts/retinal/07_eval_baselines.py --methods lad face craft ice
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
from lad.baselines.craft_train import train_craft_W  # noqa: E402
from lad.baselines.face_train import train_face_W  # noqa: E402
from lad.baselines.ice import ChannelDecompositionReducer  # noqa: E402
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
    p.add_argument("--methods", nargs="+", default=["lad", "face", "craft", "ice"])
    p.add_argument("--classes", nargs="+", default=DISEASES)
    p.add_argument("--backbone", default="densenet121")
    p.add_argument("--ckpt-suffix", default="448")
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--clip", default="biomedclip")
    p.add_argument("--grid", default="28x28")
    p.add_argument("--radius", type=int, default=8)
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--max-train", type=int, default=60)
    p.add_argument("--max-val", type=int, default=40)
    p.add_argument("--lad-epochs", type=int, default=250)
    p.add_argument("--face-iters", type=int, default=4000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "retinal" / "eval_baselines_448.csv")
    return p.parse_args()


def flat_head_for(model, backbone):
    backbone = backbone.lower()
    if backbone.startswith("resnet"):
        return model.fc
    if backbone in ("densenet121", "mobilenet_v2"):
        return model.classifier
    if backbone == "convnext_tiny":
        cls = model.classifier
        return lambda x: cls(x.unsqueeze(-1).unsqueeze(-1))
    raise NotImplementedError(backbone)


def encode_grid_A(imgs, g, input_size, device):
    """Encoder grid activations for a list of image paths -> (A=[B*L,C], hw)."""
    tf = imagenet_transform(input_size)
    chunks, hw = [], None
    with torch.inference_mode():
        for ip in imgs:
            x = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
            A_flat, _idx, hw = flatten_hw_rowmajor(g(x)[0])
            chunks.append(A_flat.cpu())
    return torch.cat(chunks, 0).to(device).float(), hw


def score_W(W, val_imgs, g, h_2d, target, input_size, device):
    """Score a basis W=[C,r] on val images with the LAD eval metrics."""
    A, hw = encode_grid_A(val_imgs, g, input_size, device)
    U = nonneg_ls_init_U_hybrid(A, W, pgd_iters=30)
    B = len(val_imgs); h, w = hw; L = h * w; rank = W.shape[1]
    with torch.inference_mode():
        A_hat = (U @ W.T).reshape(B, h, w, -1).permute(0, 3, 1, 2)
        acc_recon = float((h_2d(A_hat).argmax(-1).cpu() == target).float().mean())
    U_per = U.reshape(B, L, rank).mean(dim=1)
    importance = estimate_importance(U_per.cpu().numpy(), W.T.cpu().numpy(),
                                     h_2d, h_2d, target, 64, rank, device)
    U_sp = U.reshape(B, h, w, rank).cpu()
    W_io = W.T.cpu()
    ins = concept_insertion(U_sp, W_io, h_2d, target, importance)
    dele = concept_deletion(U_sp, W_io, h_2d, target, importance)
    return {
        "acc_recon": round(acc_recon, 4),
        "c_ins_auc": round(float(compute_insertion_auc(ins)), 4),
        "c_del_aopc": round(float(compute_deletion_score(dele)), 4),
        "gini": round(float(calculate_gini(importance)), 4),
        "sparsity": round(float(compute_sparsity(U_per)), 4),
        "n_val": B,
    }


def learn_W(method, cls, train_imgs, images_4d, g, model, flat_head, args, ckpt, classifier, bundle, dev):
    """Return a basis W=[C,r] for the requested method."""
    if method == "lad":
        grid = tuple(int(x) for x in args.grid.split("x"))
        r = probe_and_learn(cls, args.clip, train_imgs, checkpoint=ckpt,
                            concepts_dir=paths.RETINAL_CONCEPTS_DIR, device=dev, grid=grid,
                            radius=args.radius, epochs=args.lad_epochs, classifier=classifier,
                            clip_bundle=bundle, input_size=args.input_size)
        return r["W"].to(dev).float()
    if method == "face":
        W, _ = train_face_W(images_4d, model, g, flat_head, args.rank,
                            n_iter=args.face_iters, eps_early=0.0, device=dev)  # no premature stop
        return W.to(dev).float()
    if method == "craft":
        W, _ = train_craft_W(images_4d, g, args.rank)
        return W.to(dev).float()
    if method == "ice":
        A, _hw = encode_grid_A(train_imgs, g, args.input_size, dev)
        reducer = ChannelDecompositionReducer(n_components=args.rank, reduction_alg="NMF",
                                              init="nndsvda", max_iter=500)
        reducer.fit_transform(A.clamp_min(0).cpu().numpy())
        return torch.from_numpy(reducer._reducer.components_.T).float().to(dev)
    raise ValueError(method)


def main():
    args = parse_args()
    dev = args.device
    ckpt = paths.MODELS_DIR / f"best_{args.backbone}_odir{args.ckpt_suffix}.pt"
    classifier = load_classifier_checkpoint(ckpt, device=dev)
    model, backbone, classes = classifier
    g, h_2d = make_g_and_h2d(model, backbone, device=dev)
    flat_head = flat_head_for(model, backbone)
    bundle = None
    if "lad" in args.methods:
        clip_model, _, clip_tok, _ = load_clip_model(args.clip, device=dev)
        bundle = (clip_model, clip_tok)
    tf = imagenet_transform(args.input_size)
    print(f"baselines eval {ckpt.name}  methods={args.methods}  classes={args.classes}\n")

    rows = []
    for cls in args.classes:
        target = classes.index(cls)
        img_dir = paths.RETINAL_FILTERED_ROOT / backbone / "correct" / cls
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        tr_idx, va_idx = per_class_kfold(imgs, n_folds=3, fold=0, val_ratio=0.3, seed=42)
        train = [imgs[i] for i in tr_idx][: args.max_train]
        val = [imgs[i] for i in va_idx][: args.max_val]
        # preprocessed image tensor batch for the patch-based methods (FACE/CRAFT)
        need_4d = any(m in args.methods for m in ("face", "craft"))
        images_4d = (torch.stack([tf(Image.open(p).convert("RGB")) for p in train]).to(dev)
                     if need_4d else None)

        for method in args.methods:
            t0 = time.perf_counter()
            W = learn_W(method, cls, train, images_4d, g, model, flat_head,
                        args, ckpt, classifier, bundle, dev)
            m = score_W(W, val, g, h_2d, target, args.input_size, dev)
            m.update({"method": method, "class": cls, "rank": W.shape[1],
                      "elapsed_s": round(time.perf_counter() - t0, 1)})
            rows.append(m)
            print(f"[{method:<5} {cls:<22}] Acc={m['acc_recon']:.3f}  C-Ins={m['c_ins_auc']:.3f}  "
                  f"C-Del={m['c_del_aopc']:.3f}  gini={m['gini']:.3f}  ({m['elapsed_s']:.0f}s)")

    # per-method means
    print("\n" + "=" * 64 + "\nMEANS across classes:")
    summary = []
    for method in args.methods:
        mr = [r for r in rows if r["method"] == method]
        mean = {k: round(sum(r[k] for r in mr) / len(mr), 4) for k in ("acc_recon", "c_ins_auc", "c_del_aopc", "gini")}
        summary.append({"method": method, **mean})
        print(f"  {method:<6} Acc={mean['acc_recon']:.3f}  C-Ins={mean['c_ins_auc']:.3f}  "
              f"C-Del={mean['c_del_aopc']:.3f}  gini={mean['gini']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "class", "rank", "n_val", "acc_recon", "c_ins_auc", "c_del_aopc", "gini", "sparsity", "elapsed_s"]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
