"""Phase 3 — generate the retinal concept vocabulary per class (Algorithm 1, retinal domain).

For each ODIR class, runs the iterative gpt-4o-mini + CLIP-diversity loop with the
``RETINAL_FUNDUS_DOMAIN`` prompt (ophthalmoscopic findings, localizable as heatmap regions)
and a clinical prose subject (e.g. class ``amd`` -> "an eye with age-related macular
degeneration (AMD)"). Writes one TXT per class — the same format the existing pipeline reads.

Needs ``OPENAI_API_KEY``. The CLIP used here only ranks/dedups concepts; default is
BiomedCLIP (downloads ~0.4 GB on first use). Class images for the diversity prototype come
from the Phase-0 pool ``data/odir/images/<class>/``.

Usage (lad conda env):
  python scripts/retinal/03_generate_concepts.py --clip biomedclip
  python scripts/retinal/03_generate_concepts.py --classes amd glaucoma --target-size 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402
from lad.clip_concepts import (  # noqa: E402
    RETINAL_FUNDUS_DOMAIN,
    generate_class_vocabulary,
    load_clip_model,
)

# Clinical prose subjects substituted into the prompt's "color fundus photographs of {cls}".
CLASS_SUBJECT = {
    "normal": "a healthy normal eye",
    "diabetic_retinopathy": "an eye with diabetic retinopathy",
    "glaucoma": "an eye with glaucoma",
    "cataract": "an eye with a cataract",
    "amd": "an eye with age-related macular degeneration (AMD)",
    "hypertension": "an eye with hypertensive retinopathy",
    "myopia": "a pathologically myopic eye",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=paths.RETINAL_ROOT)
    p.add_argument("--classes-file", type=Path, default=None, help="default: <data-root>/classes.txt")
    p.add_argument("--classes", nargs="+", default=None, help="Subset to run (default: all in classes.txt).")
    p.add_argument("--out-dir", type=Path, default=paths.RETINAL_CONCEPTS_DIR)
    p.add_argument("--clip", default="biomedclip", help="CLIP for the diversity filter (registry name).")
    p.add_argument("--target-size", type=int, default=25)
    p.add_argument("--n-prototype-images", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if the TXT exists.")
    return p.parse_args()


def main():
    args = parse_args()
    classes_file = args.classes_file or (args.data_root / "classes.txt")
    all_classes = [c.strip() for c in classes_file.read_text(encoding="utf-8").splitlines() if c.strip()]
    classes = args.classes or all_classes
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP '{args.clip}' for concept diversity filtering ...")
    clip_model, clip_pre, clip_tok, device = load_clip_model(args.clip, device=args.device)
    print(f"  device={device}\n")

    for cls in classes:
        out_txt = args.out_dir / f"{cls}.txt"
        if out_txt.exists() and not args.overwrite:
            print(f"[skip] {cls}: {out_txt} exists (use --overwrite)")
            continue
        cls_phrase = CLASS_SUBJECT.get(cls, cls.replace("_", " "))
        img_dir = args.data_root / "images" / cls
        print(f"[{cls}] subject={cls_phrase!r}  images={img_dir}")
        vocab, stats = generate_class_vocabulary(
            cls,
            target_size=args.target_size,
            clip_model=clip_model,
            clip_preprocess=clip_pre,
            clip_tokenizer=clip_tok,
            clip_device=device,
            class_image_dir=img_dir if img_dir.exists() else None,
            domain=RETINAL_FUNDUS_DOMAIN,
            cls_phrase=cls_phrase,
        )
        out_txt.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        print(f"  -> {len(vocab)} concepts in {stats.rounds} rounds; wrote {out_txt}")
        print(f"     sample: {', '.join(vocab[:6])}\n")

    print(f"Done. Concept TXTs in {args.out_dir}")


if __name__ == "__main__":
    main()
