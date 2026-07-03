"""Phase 1 — train the retinal classifier whose decisions the concept maps will explain.

Transfer-learns a torchvision backbone (ImageNet-init encoder + fresh N-way head) on the
ODIR-5K class folders organized by ``00_prepare_odir.py``. Handles ODIR's class imbalance
with inverse-frequency-weighted cross-entropy and selects the best epoch by **macro-recall**
(balanced accuracy), which is the honest metric here (Normal/DR dominate the counts).

Saves ``models/best_<backbone>_odir.pt`` =
``{"model": state_dict, "backbone": str, "classes": [...], "val_acc", "val_balacc", "epoch"}``
— the schema ``lad.backbones.load_classifier_checkpoint`` expects.

Usage (lad conda env, GPU):
  python scripts/retinal/01_train_classifier.py --backbone resnet34 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402
from lad.backbones import build_classifier  # noqa: E402
from lad.data import IMAGENET_MEAN, IMAGENET_STD, imagenet_transform  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--data-root", type=Path, default=paths.RETINAL_ROOT)
    p.add_argument("--manifest", type=Path, default=None, help="default: <data-root>/odir_manifest.csv")
    p.add_argument("--classes-file", type=Path, default=None, help="default: <data-root>/classes.txt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Train only the new head (faster; usually lower accuracy).")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Checkpoint to load weights from before training (e.g. fine-tune a 224 model at 448).")
    p.add_argument("--out", type=Path, default=None,
                   help="default: <MODELS_DIR>/best_<backbone>_odir.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_transform(img_size: int) -> transforms.Compose:
    """Mild fundus-appropriate augmentation. No vertical flip (anatomy is up/down oriented)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    """Reads (path, class_idx) rows for one split of the ODIR manifest."""

    def __init__(self, df: pd.DataFrame, data_root: Path, transform):
        self.rows = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(self.data_root / r["path"]).convert("RGB")
        return self.transform(img), int(r["class_idx"])


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = total = 0
    per_cls_correct = np.zeros(num_classes, dtype=np.int64)
    per_cls_total = np.zeros(num_classes, dtype=np.int64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(1).cpu()
        for yi, pi in zip(y.tolist(), pred.tolist()):
            per_cls_total[yi] += 1
            if yi == pi:
                per_cls_correct[yi] += 1
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    recalls = per_cls_correct / np.maximum(per_cls_total, 1)
    return acc, float(recalls.mean()), recalls, per_cls_total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = args.manifest or (args.data_root / "odir_manifest.csv")
    classes_file = args.classes_file or (args.data_root / "classes.txt")
    classes = [c.strip() for c in classes_file.read_text(encoding="utf-8").splitlines() if c.strip()]
    num_classes = len(classes)
    out = args.out or (paths.MODELS_DIR / f"best_{args.backbone}_odir.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    tr_df, va_df = df[df.split == "train"], df[df.split == "val"]
    print(f"device={args.device} backbone={args.backbone} classes={classes}")
    print(f"train={len(tr_df)} val={len(va_df)}  -> {out}\n")

    tr_ds = ManifestDataset(tr_df, args.data_root, train_transform(args.img_size))
    va_ds = ManifestDataset(va_df, args.data_root, imagenet_transform(args.img_size))
    pin = args.device.startswith("cuda")
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin)

    # inverse-frequency class weights from the train split
    counts = Counter(tr_df["class_idx"].tolist())
    freq = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float)
    weights = (freq.sum() / (num_classes * freq.clamp(min=1))).to(args.device)
    print("train counts:", {classes[i]: int(freq[i]) for i in range(num_classes)})
    print("class weights:", {classes[i]: round(float(weights[i]), 2) for i in range(num_classes)}, "\n")

    model = build_classifier(args.backbone, num_classes, pretrained=True, device=args.device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=args.device)
        model.load_state_dict(ck.get("model", ck))
        print(f"initialized weights from {args.init_from}")
    if args.freeze_encoder:
        for n, p_ in model.named_parameters():
            if not (n.startswith("fc") or n.startswith("classifier")):
                p_.requires_grad_(False)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler(enabled=pin)

    best_balacc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n_seen = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=pin):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        sched.step()
        acc, balacc, recalls, _ = evaluate(model, va_ld, args.device, num_classes)
        tag = ""
        if balacc > best_balacc:
            best_balacc = balacc
            torch.save({"model": model.state_dict(), "backbone": args.backbone,
                        "classes": classes, "val_acc": acc, "val_balacc": balacc,
                        "epoch": epoch}, out)
            tag = "  <- saved (best)"
        history.append({"epoch": epoch, "loss": run_loss / n_seen, "val_acc": acc, "val_balacc": balacc})
        print(f"epoch {epoch:2d}/{args.epochs}  loss={run_loss/n_seen:.4f}  "
              f"val_acc={acc:.3f}  val_balacc={balacc:.3f}{tag}")

    # final per-class recall report from the best checkpoint
    best = torch.load(out, map_location=args.device)
    model.load_state_dict(best["model"])
    acc, balacc, recalls, totals = evaluate(model, va_ld, args.device, num_classes)
    print(f"\nBest epoch {best['epoch']}  val_acc={acc:.3f}  val_balacc={balacc:.3f}")
    print(f"{'class':<22}{'val_n':>7}{'recall':>9}")
    for i, c in enumerate(classes):
        print(f"{c:<22}{int(totals[i]):>7}{recalls[i]:>9.3f}")

    # always-save: metrics + per-epoch history JSON next to the checkpoint
    metrics_path = out.with_suffix(".json")
    metrics_path.write_text(json.dumps({
        "backbone": args.backbone,
        "classes": classes,
        "best_epoch": int(best["epoch"]),
        "val_acc": float(acc),
        "val_balacc": float(balacc),
        "per_class_recall": {classes[i]: float(recalls[i]) for i in range(num_classes)},
        "per_class_val_n": {classes[i]: int(totals[i]) for i in range(num_classes)},
        "train_counts": {classes[i]: int(freq[i]) for i in range(num_classes)},
        "args": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                 "weight_decay": args.weight_decay, "img_size": args.img_size,
                 "freeze_encoder": args.freeze_encoder, "seed": args.seed},
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
