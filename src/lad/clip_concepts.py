"""LAD's language pipeline: LLM concept-vocabulary generation + CLIP red-circle probing.

Two responsibilities:

1. **Vocabulary generation (Algorithm 1, supplementary §A1).** Use ``gpt-4o-mini`` under a
   strict JSON schema to propose 25 visually-grounded concepts per class, then apply lexical
   filtering and a CLIP-based diversity / class-relevance pass.
2. **Spatial CLIP probing (supplementary §A1.6).** For each filtered image, compute a
   ``[r, h, w]`` similarity tensor by drawing a thin red circle at every grid cell of a
   resize+center-crop CLIP canvas, encoding each variant, and taking cosine similarity to the
   pre-encoded concept text features. Saved as ``__P_<h>x<w>_r<radius>.npz``.

Consolidated from ``dataset_creation_pipeline.ipynb`` cells 2-11, 33-42 and the overlapping
helpers in ``otsu_based_concept_similirity_visual_prompt.ipynb``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ---------------------------------------------------------------------------
# Algorithm 1 — JSON-schema-constrained concept generation
# ---------------------------------------------------------------------------

CONCEPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "class_name": {"type": "string"},
        "concepts": {
            "type": "array",
            "minItems": 25,
            "maxItems": 25,
            "items": {
                "type": "string",
                "minLength": 3,
                "maxLength": 40,
                "pattern": r"^[a-z][a-z\-]*([ ][a-z][a-z\-]*){1,2}$",
            },
        },
    },
    "required": ["class_name", "concepts"],
}

SYSTEM_PROMPT = (
    "You generate short, visually grounded concepts for interpretability research.\n"
    "Return ONLY JSON matching the provided schema. No extra keys, no extra text."
)

GENERIC_BAN: Set[str] = {
    "animal", "mammal", "bird", "fish", "reptile", "insect",
    "object", "thing", "scene", "photo", "image", "picture", "background",
}


def humanize_class_name(s: str) -> str:
    """``diabetic_retinopathy`` -> ``diabetic retinopathy`` for prose subjects."""
    return s.replace("_", " ").strip()


@dataclass(frozen=True)
class ConceptDomain:
    """Domain-specific wording for Algorithm-1 concept generation.

    ``IMAGENET_DOMAIN`` reproduces the original ImageNet prompt byte-for-byte; other domains
    (e.g. ``RETINAL_FUNDUS_DOMAIN``) re-target the subject, examples, banned filler, the extra
    "localizable as a heatmap" guidance, and the CLIP text template.
    """
    name: str
    subject_template: str       # "{cls}" placeholder, e.g. 'the ImageNet class: "{cls}"'
    examples: str               # format-only example concepts
    clip_text_template: str     # "{c}" placeholder, e.g. "a photo of {c}"
    extra_guidance: str = ""    # appended constraint bullet lines (may be "")
    ban: frozenset = frozenset()
    humanize: bool = False      # humanize the class name before substitution


IMAGENET_DOMAIN = ConceptDomain(
    name="imagenet",
    subject_template='the ImageNet class: "{cls}"',
    examples='"striped fur", "sharp teeth", "wooden handle", "metallic surface"',
    clip_text_template="a photo of {c}",
)

RETINAL_FUNDUS_DOMAIN = ConceptDomain(
    name="retinal_fundus",
    subject_template="color fundus photographs of {cls}",
    examples='"hard exudates", "cotton-wool spots", "dot-blot hemorrhages", "optic disc cupping"',
    clip_text_template="a fundus photograph showing {c}",
    extra_guidance=(
        "- Concepts MUST be ophthalmoscopic findings or fundus landmarks a clinician could point "
        "to, and plausibly localizable as a heatmap region (lesions, vessel features, optic-disc / "
        "macula appearance, texture or color cues).\n"
        "- Prefer concrete fundus signs over diagnoses, patient attributes, or imaging artifacts.\n"
    ),
    ban=frozenset({
        "eye", "eyes", "retina", "retinal", "fundus", "ophthalmic", "ocular", "vision",
        "image", "photo", "photograph", "scan", "camera", "exposure", "patient",
        "diagnosis", "disease", "color", "colour", "circle", "red circle",
    }),
    humanize=True,
)


def build_user_prompt(
    class_name: str,
    banned: List[str],
    *,
    n: int = 25,
    domain: ConceptDomain = IMAGENET_DOMAIN,
    cls_phrase: Optional[str] = None,
) -> str:
    banned_str = ", ".join(banned[:200]) if banned else "(none)"
    cls = cls_phrase or (humanize_class_name(class_name) if domain.humanize else class_name)
    subject = domain.subject_template.format(cls=cls)
    return (
        f'Generate EXACTLY {n} distinct visual concepts for {subject}.\n\n'
        "Hard constraints:\n"
        f"- Output EXACTLY {n} concepts.\n"
        "- Each concept MUST be 2-3 words, all lowercase.\n"
        "- Concepts MUST be visually grounded (parts, textures, shapes, contexts, backgrounds, actions).\n"
        "- Do NOT use the class name or its component words; avoid trivial synonyms.\n"
        "- Avoid near-duplicates and generic filler.\n"
        f"{domain.extra_guidance}"
        f"- Avoid these banned/previous concepts: {banned_str}\n\n"
        f'Examples (format only): {domain.examples}\n\n'
        "Return JSON only."
    ).strip()


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z\s\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_2_3_words(s: str) -> bool:
    return 2 <= len(s.split()) <= 3


def class_tokens(class_name: str) -> Set[str]:
    toks: List[str] = []
    for t in normalize(class_name).split():
        toks.extend(t.split("-"))
    return {t for t in toks if t}


def too_close_to_class(concept: str, class_name: str) -> bool:
    """True if any token of ``concept`` overlaps a token of the class name."""
    c_tokens = set(concept.split())
    return bool(c_tokens & class_tokens(class_name))


@dataclass
class GenStats:
    class_name: str
    rounds: int = 0
    kept_total: int = 0
    filtered_generic: int = 0
    filtered_close: int = 0
    filtered_wordcount: int = 0
    filtered_duplicate: int = 0


def lexical_filter(
    raw: Iterable[str], class_name: str, seen: Set[str], stats: GenStats,
    *, ban: Optional[Set[str]] = None,
) -> List[str]:
    """Stage-1 (rule-based) filter from supplementary §A1.3."""
    ban = GENERIC_BAN if ban is None else ban
    kept: List[str] = []
    for c in raw:
        c2 = normalize(c)
        if not is_2_3_words(c2):
            stats.filtered_wordcount += 1
            continue
        if any(w in ban for w in c2.split()):
            stats.filtered_generic += 1
            continue
        if too_close_to_class(c2, class_name):
            stats.filtered_close += 1
            continue
        if c2 in seen:
            stats.filtered_duplicate += 1
            continue
        kept.append(c2)
        seen.add(c2)
    return kept


def call_openai_25(
    class_name: str, banned: List[str], model: str = "gpt-4o-mini", client=None,
    *, domain: ConceptDomain = IMAGENET_DOMAIN, cls_phrase: Optional[str] = None,
) -> List[str]:
    """One LLM call returning exactly 25 concept strings.

    Requires ``OPENAI_API_KEY`` in the environment. Pass ``client`` to reuse a session;
    otherwise a fresh ``openai.OpenAI()`` instance is created. ``domain`` selects the prompt
    wording (ImageNet vs. retinal fundus, etc.).
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(class_name, banned, domain=domain, cls_phrase=cls_phrase)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "concept_list",
                "strict": True,
                "schema": CONCEPT_SCHEMA,
            }
        },
    )
    return json.loads(resp.output_text)["concepts"]


# ---------------------------------------------------------------------------
# CLIP loading + text/image encoding helpers
# ---------------------------------------------------------------------------


# Short-name registry for the CLIP backbones we probe with. Each maps to an
# ``(model_name, pretrained)`` pair for ``open_clip.create_model_and_transforms``.
# hf-hub model ids carry their own weights, so ``pretrained`` is None there.
CLIP_REGISTRY = {
    "laion": ("ViT-B-16", "laion2b_s34b_b88k"),          # current ImageNet-pipeline default
    "openai": ("ViT-B-16", "openai"),                     # OpenAI WIT pretraining
    "biomedclip": ("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", None),
}


def resolve_clip(name: str) -> Tuple[str, Optional[str]]:
    """Map a short registry name (``"biomedclip"``) to ``(model_name, pretrained)``.

    Pass-through for anything not in the registry (already a full open_clip model name).
    """
    return CLIP_REGISTRY.get(name.lower(), (name, None if name.startswith("hf-hub:") else "openai"))


def load_fairclip(checkpoint: str, device: str, base_arch: str = "ViT-B/16"):
    """Load FairCLIP: OpenAI CLIP ``base_arch`` with a fundus-finetuned ``model_state_dict``.

    FairCLIP checkpoints are plain OpenAI-CLIP state dicts, so we load via the ``clip`` package
    (key format matches) and overwrite the weights. Returns ``(model, preprocess, tokenizer)``.
    The model is cast to fp32 for compatibility with the red-circle probing pipeline.
    """
    import clip  # openai clip (clip-anytorch); key format matches FairCLIP checkpoints

    model, preprocess = clip.load(base_arch, device=device, jit=False)
    model = model.float()
    if checkpoint:
        ck = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()
    return model, preprocess, clip.tokenize


def load_clip_model(
    model_name: str = "ViT-B-16",
    pretrained: Optional[str] = "laion2b_s34b_b88k",
    device: Optional[str] = None,
    checkpoint: Optional[str] = None,
):
    """Load a CLIP model. Returns ``(model, preprocess, tokenizer, device)``.

    ``model_name`` may be a registry short-name (``"biomedclip"``, ``"laion"``, ``"openai"``,
    ``"fairclip"``), a full open_clip architecture (``"ViT-B-16"``), or an ``"hf-hub:org/model"``
    id. ``"fairclip"`` requires ``checkpoint`` (path to the fundus-finetuned .pth) and loads via
    the OpenAI ``clip`` package. For registry names and hf-hub ids ``pretrained`` is auto-resolved.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name.lower() == "fairclip":
        if not checkpoint:
            raise ValueError("load_clip_model('fairclip') requires checkpoint=<path to FairCLIP .pth>.")
        model, preprocess, tokenizer = load_fairclip(checkpoint, device)
        return model, preprocess, tokenizer, device
    if model_name.lower() in CLIP_REGISTRY:
        model_name, pretrained = CLIP_REGISTRY[model_name.lower()]
    elif model_name.startswith("hf-hub:"):
        pretrained = None  # hf-hub checkpoints carry their own weights
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        return model, preprocess, tokenizer, device
    except ImportError:
        import clip  # type: ignore

        official = model_name.replace("-", "/")
        model, preprocess = clip.load(official, device=device, jit=False)
        model.eval()
        return model, preprocess, clip.tokenize, device


def _tokenize_text(prompts: List[str], tokenizer):
    return tokenizer(prompts)


@torch.no_grad()
def clip_encode_texts(prompts: List[str], model, tokenizer, device: str) -> torch.Tensor:
    tokens = _tokenize_text(prompts, tokenizer).to(device)
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


@torch.no_grad()
def clip_encode_images(image_paths: List[Path], model, preprocess, device: str, batch_size: int = 32) -> torch.Tensor:
    feats = []
    for i in range(0, len(image_paths), batch_size):
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in image_paths[i : i + batch_size]]
        x = torch.stack(imgs).to(device)
        feats.append(F.normalize(model.encode_image(x).float(), dim=-1))
    return torch.cat(feats, dim=0)


def encode_text_features(
    prompts: List[str],
    model,
    tokenizer,
    device: str,
    cache_path: Optional[Path] = None,
) -> torch.Tensor:
    """Cache concept text features per class for reuse across many images."""
    if cache_path is not None and cache_path.exists():
        obj = torch.load(cache_path, map_location=device)
        return obj["text_features"]
    tf = clip_encode_texts(prompts, model, tokenizer, device)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prompts": prompts, "text_features": tf}, cache_path)
    return tf


# ---------------------------------------------------------------------------
# CLIP-based concept filter (supplementary §A1.4)
# ---------------------------------------------------------------------------


def filter_concepts_clip(
    candidates: List[str],
    class_name: str,
    model,
    preprocess,
    tokenizer,
    device: str,
    class_image_dir: Optional[Path] = None,
    n_images: int = 20,
    max_sim_between: float = 0.80,
    use_image_ranking: bool = True,
    seed: int = 0,
    text_template: str = "a photo of {c}",
) -> List[str]:
    """Stage-2 CLIP filter: rank by similarity to the class image prototype, then dedup."""
    if not candidates:
        return []
    concept_emb = clip_encode_texts([text_template.format(c=c) for c in candidates], model, tokenizer, device)

    scores: Optional[torch.Tensor] = None
    if use_image_ranking and class_image_dir is not None and Path(class_image_dir).exists():
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        img_paths = [p for p in Path(class_image_dir).rglob("*") if p.suffix.lower() in exts]
        if img_paths:
            import random

            rnd = random.Random(seed)
            rnd.shuffle(img_paths)
            img_paths = img_paths[: min(n_images, len(img_paths))]
            img_emb = clip_encode_images(img_paths, model, preprocess, device, batch_size=32)
            mu = F.normalize(img_emb.mean(dim=0, keepdim=True), dim=-1)
            scores = (concept_emb @ mu.T).squeeze(1)

    order = torch.argsort(scores, descending=True).tolist() if scores is not None else list(range(len(candidates)))

    kept: List[str] = []
    kept_embs: List[torch.Tensor] = []
    for idx in order:
        c = candidates[idx]
        e = concept_emb[idx]
        if not kept_embs:
            kept.append(c)
            kept_embs.append(e)
            continue
        sims = torch.stack(kept_embs) @ e.unsqueeze(-1)
        if float(sims.max()) >= max_sim_between:
            continue
        kept.append(c)
        kept_embs.append(e)
    return kept


# ---------------------------------------------------------------------------
# Algorithm 1 — full iterative loop
# ---------------------------------------------------------------------------


def generate_class_vocabulary(
    class_name: str,
    *,
    model_name: str = "gpt-4o-mini",
    max_rounds: int = 10,
    extra_fallback_rounds: int = 5,
    target_size: int = 25,
    clip_model=None,
    clip_preprocess=None,
    clip_tokenizer=None,
    clip_device: str = "cuda",
    class_image_dir: Optional[Path] = None,
    openai_client=None,
    domain: ConceptDomain = IMAGENET_DOMAIN,
    cls_phrase: Optional[str] = None,
) -> Tuple[List[str], GenStats]:
    """Run Algorithm 1 to produce ``target_size`` filtered concepts for ``class_name``.

    ``domain`` selects prompt wording + banned filler + CLIP text template (ImageNet vs.
    retinal fundus). ``cls_phrase`` overrides the prose name used in the prompt subject
    (e.g. ``"age-related macular degeneration"`` for class ``amd``).
    """
    stats = GenStats(class_name=class_name)
    seen: Set[str] = set()
    kept: List[str] = []
    banned: List[str] = []
    ban = GENERIC_BAN | set(domain.ban)

    for r in range(max_rounds):
        stats.rounds += 1
        raw = call_openai_25(class_name, banned, model=model_name, client=openai_client,
                             domain=domain, cls_phrase=cls_phrase)
        use_clip_pass = (r < max_rounds - 1) and clip_model is not None
        stage1 = lexical_filter(raw, class_name, seen, stats, ban=ban)
        new_kept = (
            filter_concepts_clip(
                stage1, class_name, clip_model, clip_preprocess, clip_tokenizer, clip_device,
                class_image_dir=class_image_dir, text_template=domain.clip_text_template,
            )
            if use_clip_pass
            else stage1
        )
        kept.extend(new_kept)
        stats.kept_total = len(kept)
        banned = list(seen)
        if len(kept) >= target_size:
            return kept[:target_size], stats

    # fallback rounds: lexical-only
    for _ in range(extra_fallback_rounds):
        raw = call_openai_25(class_name, banned, model=model_name, client=openai_client,
                             domain=domain, cls_phrase=cls_phrase)
        kept.extend(lexical_filter(raw, class_name, seen, stats, ban=ban))
        banned = list(seen)
        if len(kept) >= target_size:
            return kept[:target_size], stats
    raise RuntimeError(f"Could not collect {target_size} concepts for class={class_name!r}")


def load_concepts_txt(path: Path | str) -> List[str]:
    """Read a per-class concept TXT (newline-separated). Skips empty / comment lines."""
    seen, concepts = set(), []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if s not in seen:
                seen.add(s)
                concepts.append(s)
    return concepts


# ---------------------------------------------------------------------------
# Spatial red-circle CLIP probing (supplementary §A1.6, paper Eq. building S)
# ---------------------------------------------------------------------------


def draw_red_circle(img: Image.Image, center: Tuple[int, int], radius: int, line_width: int = 2) -> Image.Image:
    """Draw a thin red circle outline at ``center``. Returns a *new* image."""
    x, y = center
    out = img.copy()
    ImageDraw.Draw(out).ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 0, 0), width=line_width)
    return out


def grid_centers(H: int, W: int, Ht: int, Wt: int) -> List[Tuple[int, int]]:
    """Return ``Ht*Wt`` grid-cell centers in pixel coordinates."""
    centers: List[Tuple[int, int]] = []
    step_y, step_x = H / Ht, W / Wt
    off_y, off_x = step_y / 2, step_x / 2
    for r in range(Ht):
        for c in range(Wt):
            centers.append((int(c * step_x + off_x), int(r * step_y + off_y)))
    return centers


def _make_clip_geom_post():
    """Deterministic CLIP-style preprocessing: resize→center-crop, then to-tensor + normalize."""
    geom = transforms.Compose(
        [transforms.Resize(224, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224)]
    )
    post = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return geom, post


@torch.no_grad()
def compute_P_for_image(
    img_path: Path,
    text_features: torch.Tensor,
    concepts: List[str],
    model,
    device: str,
    grid_size: Tuple[int, int] = (14, 14),
    radius: int = 16,
    line_width: int = 2,
    batch_size: int = 196,
    use_logit_scale: bool = True,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int]]:
    """Build the spatial similarity tensor ``P = [M, Ht, Wt]`` for one image.

    The red circle is drawn *after* CLIP geometry so probing reflects exactly what CLIP sees.
    """
    img0 = Image.open(img_path).convert("RGB")
    geom, post = _make_clip_geom_post()
    img = geom(img0)
    W_pix, H_pix = img.size

    Ht, Wt = grid_size
    centers = grid_centers(H_pix, W_pix, Ht, Wt)

    embeds = []
    amp_on = device.startswith("cuda")
    with torch.cuda.amp.autocast(enabled=amp_on):
        batch: List[torch.Tensor] = []
        for c in centers:
            batch.append(post(draw_red_circle(img, c, radius=radius, line_width=line_width)))
            if len(batch) == batch_size:
                feats = model.encode_image(torch.stack(batch).to(device))
                embeds.append(F.normalize(feats.float(), dim=-1))
                batch = []
        if batch:
            feats = model.encode_image(torch.stack(batch).to(device))
            embeds.append(F.normalize(feats.float(), dim=-1))
        image_embeds = torch.cat(embeds, dim=0)
        S = image_embeds @ text_features.T
        if use_logit_scale and hasattr(model, "logit_scale"):
            S = S * model.logit_scale.exp()
    P = S.view(Ht, Wt, len(concepts)).permute(2, 0, 1).contiguous()
    return P.cpu(), centers, (H_pix, W_pix)


@torch.no_grad()
def compute_P_tiled(
    img_path: Path,
    text_features: torch.Tensor,
    concepts: List[str],
    model,
    device: str,
    canvas: int = 672,
    grid_size: Tuple[int, int] = (14, 14),
    window: int = 224,
    batch_size: int = 64,
    use_logit_scale: bool = True,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int]]:
    """Sliding-window CLIP probing -> higher-resolution ``P = [M, Ht, Wt]``.

    Instead of one global 224 view with a red circle, resize the fundus to a larger ``canvas``
    and encode a zoomed ``window``x``window`` crop centered on each grid cell (each crop resized
    to CLIP's 224). Every cell's embedding therefore sees its local region at full 224 detail,
    lifting effective resolution past the global-224 ceiling that the red-circle prober hits.
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    img = transforms.Compose([
        transforms.Resize(canvas, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(canvas),
    ])(Image.open(img_path).convert("RGB"))
    post = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    Ht, Wt = grid_size
    centers = grid_centers(canvas, canvas, Ht, Wt)
    half = window // 2
    patches = []
    for (cx, cy) in centers:
        left = max(0, min(cx - half, canvas - window))
        top = max(0, min(cy - half, canvas - window))
        patches.append(post(img.crop((left, top, left + window, top + window))))

    embeds = []
    with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        for i in range(0, len(patches), batch_size):
            feats = model.encode_image(torch.stack(patches[i:i + batch_size]).to(device))
            embeds.append(F.normalize(feats.float(), dim=-1))
        image_embeds = torch.cat(embeds, dim=0)
        S = image_embeds @ text_features.T
        if use_logit_scale and hasattr(model, "logit_scale"):
            S = S * model.logit_scale.exp()
    P = S.view(Ht, Wt, len(concepts)).permute(2, 0, 1).contiguous()
    return P.cpu(), centers, (canvas, canvas)


def save_P_npz(
    out_file: Path,
    P: torch.Tensor,
    concepts: List[str],
    centers: List[Tuple[int, int]],
    grid_size: Tuple[int, int],
    radius: int,
    image_hw: Tuple[int, int],
    image_path: Path,
    importance_scores: Optional[torch.Tensor] = None,
    otsu_masks: Optional[torch.Tensor] = None,
) -> None:
    """Persist a per-image P-tensor in the canonical NPZ schema."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        P=P.numpy(),
        concepts=np.array(concepts, dtype=object),
        centers=np.array(centers, dtype=np.int32),
        grid_size=np.array(grid_size, dtype=np.int32),
        radius=np.array([radius], dtype=np.int32),
        image_hw=np.array(list(image_hw), dtype=np.int32),
        image_path=np.array([str(image_path)], dtype=object),
    )
    if importance_scores is not None:
        payload["importance_scores"] = importance_scores.numpy()
    if otsu_masks is not None:
        payload["otsu_masks"] = otsu_masks.numpy()
    np.savez_compressed(out_file.as_posix(), **payload)


# ---------------------------------------------------------------------------
# Otsu post-processing (used to rank concepts during NPZ creation)
# ---------------------------------------------------------------------------


def _otsu_threshold(scores_np: np.ndarray) -> float:
    mn, mx = float(scores_np.min()), float(scores_np.max())
    if mx == mn:
        return mn
    scaled = ((scores_np - mn) / (mx - mn) * 255.0).astype(np.uint8)
    hist, _ = np.histogram(scaled, bins=256, range=(0, 256))
    total = scaled.size
    cdf = np.cumsum(hist)
    cdf_m = np.cumsum(hist * np.arange(256))
    best_var, best_t = -1.0, 0
    for t in range(256):
        if cdf[t] == 0 or cdf[t] == total:
            continue
        w1 = cdf[t] / total
        w2 = 1.0 - w1
        m1 = cdf_m[t] / (cdf[t] + 1e-9)
        m2 = (cdf_m[-1] - cdf_m[t]) / ((total - cdf[t]) + 1e-9)
        var = w1 * w2 * (m1 - m2) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return float(mn + (best_t / 255.0) * (mx - mn))


def calculate_importance_and_masks(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-concept Otsu mask + mean activation inside the mask."""
    P_cpu = P.detach().float().cpu().numpy()
    M, Ht, Wt = P_cpu.shape
    imp = np.zeros((M,), dtype=np.float32)
    masks = np.zeros((M, Ht, Wt), dtype=np.bool_)
    for m in range(M):
        smap = P_cpu[m]
        thr = _otsu_threshold(smap)
        mask = smap >= thr
        masks[m] = mask
        if mask.any():
            imp[m] = float((smap * mask).sum() / mask.sum())
    return torch.from_numpy(imp), torch.from_numpy(masks)


__all__ = [
    # Algorithm 1
    "CONCEPT_SCHEMA",
    "SYSTEM_PROMPT",
    "GENERIC_BAN",
    "ConceptDomain",
    "IMAGENET_DOMAIN",
    "RETINAL_FUNDUS_DOMAIN",
    "humanize_class_name",
    "build_user_prompt",
    "normalize",
    "is_2_3_words",
    "class_tokens",
    "too_close_to_class",
    "GenStats",
    "lexical_filter",
    "filter_concepts_clip",
    "call_openai_25",
    "generate_class_vocabulary",
    "load_concepts_txt",
    # CLIP
    "CLIP_REGISTRY",
    "resolve_clip",
    "load_clip_model",
    "load_fairclip",
    "clip_encode_texts",
    "clip_encode_images",
    "encode_text_features",
    # Red-circle probing
    "draw_red_circle",
    "grid_centers",
    "compute_P_for_image",
    "compute_P_tiled",
    "save_P_npz",
    # Otsu
    "calculate_importance_and_masks",
]
