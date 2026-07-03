"""Step 3: generate the per-class concept vocabulary via gpt-4o-mini + CLIP filter.

For each class in ``--classes-file`` that does NOT yet have a TXT under
``concept_outputs/per_class_txt/<class>.txt``, we run Algorithm 1 of the supplementary
material and write the resulting 25 concepts.

OpenAI rate-limit handling is intentionally simple: failures are logged and retried with
exponential backoff inside the OpenAI SDK; this script does not parallelize across
classes (the LLM-side wait dominates and OpenAI per-account RPM caps are easy to hit).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # type: ignore  # noqa: E402
from lad.clip_concepts import generate_class_vocabulary, load_clip_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classes-file", type=Path, default=paths.CLASSES_FILE)
    p.add_argument("--concepts-dir", type=Path, default=paths.CONCEPTS_DIR)
    p.add_argument("--filtered-root", type=Path, default=paths.FILTERED_ROOT,
                   help="Used to seed the CLIP image-prototype filter. Pass --no-image-rank to skip.")
    p.add_argument("--prototype-backbone", default="resnet34",
                   help="Which backbone's filtered images to use for the CLIP image-prototype filter.")
    p.add_argument("--no-image-rank", action="store_true",
                   help="Skip the CLIP image-prototype ranking (lexical + text-dedup only).")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--clip-model", default="ViT-B-16")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sleep-between", type=float, default=0.4,
                   help="Seconds to sleep between LLM calls (rate-limit cushion).")
    p.add_argument("--num-concepts", type=int, default=25,
                   help="Target concepts per class (passed to generate_class_vocabulary as target_size).")
    return p.parse_args()


def main():
    args = parse_args()
    args.concepts_dir.mkdir(parents=True, exist_ok=True)
    classes = [ln.strip() for ln in args.classes_file.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
    print(f"Loaded {len(classes)} classes from {args.classes_file}")

    if args.no_image_rank:
        clip_bundle = (None, None, None, args.device)
    else:
        print("Loading CLIP for filtering ...")
        clip_bundle = load_clip_model(args.clip_model, pretrained=args.clip_pretrained, device=args.device)

    clip_model, clip_preprocess, clip_tokenizer, clip_device = clip_bundle

    pending = [c for c in classes if not (args.concepts_dir / f"{c}.txt").exists()]
    print(f"{len(classes) - len(pending)} already done, {len(pending)} to generate.")

    for i, cls in enumerate(pending, 1):
        out_file = args.concepts_dir / f"{cls}.txt"
        cls_image_dir = args.filtered_root / args.prototype_backbone / "correct" / cls if not args.no_image_rank else None
        try:
            vocab, stats = generate_class_vocabulary(
                cls,
                model_name=args.openai_model,
                target_size=args.num_concepts,
                max_rounds=args.max_rounds,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
                clip_device=clip_device,
                class_image_dir=cls_image_dir,
            )
            out_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
            print(f"[{i}/{len(pending)}] {cls!r}: kept {len(vocab)} after {stats.rounds} rounds")
        except Exception as exc:
            print(f"[{i}/{len(pending)}] {cls!r}: FAILED ({exc!r})")
        time.sleep(args.sleep_between)

    print("Concept generation complete.")


if __name__ == "__main__":
    main()
