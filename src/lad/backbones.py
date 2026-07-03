"""Pretrained image classifiers exposed as a feature-extractor / classifier-head pair.

Every backbone resolves to two callables:

* ``g(x: [B, 3, 224, 224]) -> (z: [B, C, h, w], side: Optional[Tensor])``
* ``h_2d(z: [B, C, h, w], side: Optional[Tensor] = None) -> [B, num_classes]``

For convolutional backbones ``side`` is always ``None`` — the spatial map ``z`` carries
all of the encoder's state, and ``h_2d`` is just GAP + linear head. For ``vit_b_16``,
``side`` carries the **[CLS] token at the tap point** (post-block-10, before block-11).
``h_2d`` then re-attaches the CLS, runs the model's final transformer block + LayerNorm,
extracts the CLS, and applies the original classification head — *no extra training*.
This is the "penultimate-layer tap, finish the forward pass" scheme: the model's own
weights and head are unchanged; the only thing different is that block 11's input
patches have been replaced by their concept reconstruction.

Concrete encoder shapes (ImageNet-pretrained):

==============   ===============   ===========   ==================
backbone          C (channels)      (h, w)        side state shape
==============   ===============   ===========   ==================
resnet18          512               (7, 7)        None
resnet34          512               (7, 7)        None
resnet50          2048              (7, 7)        None
mobilenet_v2      1280              (7, 7)        None
convnext_tiny     768               (7, 7)        None
vit_b_16          768               (14, 14)      [B, 1, 768]  (CLS at tap)
==============   ===============   ===========   ==================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

# g(x) -> (z_4d, side_state); side_state is Optional[Tensor]
G = Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
# h_2d(z_4d, side_state=None) -> logits
H2D = Callable[..., torch.Tensor]

_SUPPORTED = {
    "resnet18", "resnet34", "resnet50", "mobilenet_v2", "convnext_tiny", "vit_b_16",
    # Places365 — non-ImageNet backbone (365-class scene classifier, sub-90 saturation).
    "resnet50_places365",
    # HAM10000 — locally-trained ConvNeXt-Small dermoscopy classifier (7-way). Resolved
    # from a checkpoint under MODELS_DIR (best_<name>.pt), not torchvision. The _448 variant
    # is the 448px fine-tune (14x14 encoder grid vs 7x7).
    "convnext_small_ham10000",
    "convnext_small_ham10000_448",
}


def _models_dir() -> Path:
    """Repo ``models/`` dir (override via ``LAD_MODELS_DIR``). Avoids importing paths.py
    here so backbones stays import-light; resolves relative to this file otherwise."""
    env = os.environ.get("LAD_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models"

# CSAILVision pretrained Places365 weights (canonical source, ~98 MB).
_PLACES365_RESNET50_URL = (
    "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
)


def _load_places365_resnet50() -> nn.Module:
    """Load ResNet-50 with Places365 pretrained weights (365-class output)."""
    model = torchvision.models.resnet50(num_classes=365)
    state = torch.hub.load_state_dict_from_url(
        _PLACES365_RESNET50_URL,
        map_location="cpu",
        file_name="resnet50_places365.pth.tar",
        check_hash=False,
    )
    # CSAILVision checkpoints wrap weights in DataParallel: keys start with "module.".
    sd = {k.replace("module.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(sd)
    return model


def load_backbone(name: str, device: str = "cuda") -> nn.Module:
    """Load a pretrained backbone, frozen and ``.eval()``-ed.

    All entries are ImageNet-pretrained except ``resnet50_places365`` which is the MIT
    Places365-Standard scene classifier.
    """
    name = name.lower()
    if name not in _SUPPORTED:
        raise ValueError(f"Unsupported backbone {name!r}. Choose from {sorted(_SUPPORTED)}.")

    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
    elif name == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        model = torchvision.models.resnet34(weights=weights)
    elif name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
    elif name == "mobilenet_v2":
        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
        model = torchvision.models.mobilenet_v2(weights=weights)
    elif name == "convnext_tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        model = torchvision.models.convnext_tiny(weights=weights)
    elif name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
    elif name.startswith("convnext_small_ham10000"):
        # Locally-trained 7-way dermoscopy head (base or _448 fine-tune). Checkpoint name
        # mirrors the backbone name. Reuse the checkpoint loader, then return directly.
        ckpt = _models_dir() / f"best_{name}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} not found — train it first with scripts/train_ham_classifier.py"
            )
        model, _bb, _classes = load_classifier_checkpoint(ckpt, device=device, freeze=True)
        return model  # already frozen + eval on device
    else:  # resnet50_places365
        model = _load_places365_resnet50()

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Custom-head classifiers (for non-ImageNet label spaces, e.g. retinal ODIR-5K).
# These keep the torchvision encoder but swap the
# final Linear to `num_classes`, so `make_g_and_h2d` below works unchanged once
# the head has been replaced.
# ---------------------------------------------------------------------------

_CLASSIFIER_CTORS = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights),
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights),
    "mobilenet_v2": (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights),
    "convnext_tiny": (torchvision.models.convnext_tiny, torchvision.models.ConvNeXt_Tiny_Weights),
    "convnext_small": (torchvision.models.convnext_small, torchvision.models.ConvNeXt_Small_Weights),
    "densenet121": (torchvision.models.densenet121, torchvision.models.DenseNet121_Weights),
}


def _replace_head(model: nn.Module, backbone: str, num_classes: int) -> nn.Module:
    """Swap the final classification Linear for an ``num_classes``-way head, in place."""
    backbone = backbone.lower()
    if backbone.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone in ("mobilenet_v2", "convnext_tiny", "convnext_small"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"_replace_head: unsupported backbone {backbone!r}")
    return model


def build_classifier(
    name: str, num_classes: int, *, pretrained: bool = True, device: str = "cuda"
) -> nn.Module:
    """A torchvision backbone with its head replaced by an ``num_classes``-way Linear.

    ``pretrained=True`` initializes the encoder from ImageNet weights (transfer learning for
    a new domain such as fundus images); the new head is always randomly initialized.
    Returned model is **trainable** and on ``device`` (caller decides freeze/eval).
    """
    name = name.lower()
    if name not in _CLASSIFIER_CTORS:
        raise ValueError(f"build_classifier: unsupported backbone {name!r}. "
                         f"Choose from {sorted(_CLASSIFIER_CTORS)}.")
    ctor, weights_enum = _CLASSIFIER_CTORS[name]
    model = ctor(weights=weights_enum.DEFAULT if pretrained else None)
    model = _replace_head(model, name, num_classes)
    return model.to(device)


def load_classifier_checkpoint(
    checkpoint_path, *, device: str = "cuda", freeze: bool = True
) -> Tuple[nn.Module, str, list]:
    """Load a checkpoint saved by ``scripts/retinal/01_train_classifier.py``.

    Checkpoint schema: ``{"model": state_dict, "backbone": str, "classes": list[str], ...}``.
    Returns ``(model, backbone, classes)``. By default the model is frozen and ``.eval()``-ed
    for inference (filtering / LAD / eval). ``classes`` order defines the head index order.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "backbone" not in ckpt or "classes" not in ckpt:
        raise KeyError(f"{checkpoint_path}: checkpoint missing 'backbone'/'classes' keys.")
    backbone, classes = ckpt["backbone"], list(ckpt["classes"])
    state = ckpt.get("model", ckpt)
    model = build_classifier(backbone, len(classes), pretrained=False, device=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model, backbone, classes


def make_g_and_h2d(
    model: nn.Module,
    backbone: str,
    device: str = "cuda",
    vit_cls_mode: str = "preserve",
) -> Tuple[G, H2D]:
    """Split ``model`` into ``(g, h_2d)`` matching the contract documented at module top.

    Parameters
    ----------
    vit_cls_mode : applies only to ``vit_b_16``. Two interpretations of Plan B:
        - ``"preserve"`` (default, "interpretation (a)"): keep the [CLS] token computed
          from the original patches through the first 11 blocks. The classifier head sees
          a CLS that already encodes the full image; concepts only influence the marginal
          update from block 12.
        - ``"mean"`` ("interpretation (b)"): overwrite the [CLS] token at the tap point
          with the spatial mean of the (possibly reconstructed) patch tokens. Concepts
          then drive both block 12's attention *and* its CLS update — single-layer
          deletion has full effect on the prediction.
    """
    backbone = backbone.lower()
    if vit_cls_mode not in ("preserve", "mean"):
        raise ValueError(f"vit_cls_mode must be 'preserve' or 'mean', got {vit_cls_mode!r}")

    if backbone.startswith("resnet"):
        encoder = nn.Sequential(*list(model.children())[:-2]).eval().to(device)
        fc = model.fc.eval()

        def g(x: torch.Tensor):
            return encoder(x), None

        def h_2d(z: torch.Tensor, side: Optional[torch.Tensor] = None) -> torch.Tensor:
            return fc(z.mean(dim=(2, 3)))

        return g, h_2d

    if backbone == "mobilenet_v2":
        encoder = model.features.eval().to(device)
        classifier = model.classifier.eval()

        def g(x: torch.Tensor):
            return encoder(x), None

        def h_2d(z: torch.Tensor, side: Optional[torch.Tensor] = None) -> torch.Tensor:
            return classifier(z.mean(dim=(2, 3)))

        return g, h_2d

    if backbone.startswith("convnext"):  # convnext_tiny / convnext_small / convnext_small_ham10000
        encoder = model.features.eval().to(device)
        classifier = model.classifier.eval()  # Sequential: LayerNorm2d, Flatten, Linear

        def g(x: torch.Tensor):
            return encoder(x), None

        def h_2d(z: torch.Tensor, side: Optional[torch.Tensor] = None) -> torch.Tensor:
            pooled = z.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            return classifier(pooled)

        return g, h_2d

    if backbone == "densenet121":
        # torchvision densenet forward = features -> relu -> GAP -> classifier(Linear).
        # Fold the relu into the encoder so activations are non-negative (like resnet/
        # mobilenet), suiting the W>=0 reconstruction A ~= S Wᵀ. encoder -> [B, 1024, 7, 7].
        encoder = nn.Sequential(model.features, nn.ReLU(inplace=True)).eval().to(device)
        classifier = model.classifier.eval()

        def g(x: torch.Tensor):
            return encoder(x), None

        def h_2d(z: torch.Tensor, side: Optional[torch.Tensor] = None) -> torch.Tensor:
            return classifier(z.mean(dim=(2, 3)))

        return g, h_2d

    if backbone == "vit_b_16":
        # Plan B: tap after block index 10 (= 11 of 12 blocks), reconstruct patch tokens
        # in concept space, then let block 11 + final LN + heads finish the forward pass.
        # The model's weights and head are unchanged. Only the patch tokens entering the
        # final block are restricted to the concept reconstruction; the CLS at the tap
        # point is preserved (interpretation (a) — counterfactual at the last layer).
        encoder = model.encoder
        layers = encoder.layers              # nn.Sequential of 12 EncoderBlocks
        if len(layers) != 12:
            raise RuntimeError(f"vit_b_16 expected 12 blocks, got {len(layers)}")
        pos_embedding = encoder.pos_embedding
        encoder_dropout = encoder.dropout
        final_ln = encoder.ln.eval()
        heads = model.heads.eval()
        cls_token_param = model.class_token   # [1, 1, 768]
        process_input = model._process_input  # patch_embed → [B, 196, 768]

        # Pre-pull the first-11 blocks and the last (12th) block as separate sub-modules.
        first_11 = nn.Sequential(*list(layers)[:11]).eval().to(device)
        last_block = layers[11].eval().to(device)

        def g(x: torch.Tensor):
            # 1. Patch-embed: [B, 3, 224, 224] -> [B, 196, 768]
            patches_input = process_input(x)
            n = patches_input.shape[0]
            # 2. Prepend CLS token, add positional embedding, dropout
            cls = cls_token_param.expand(n, -1, -1)
            seq = torch.cat([cls, patches_input], dim=1)  # [B, 197, 768]
            seq = encoder_dropout(seq + pos_embedding)
            # 3. Run the first 11 transformer blocks
            seq = first_11(seq)  # [B, 197, 768] post-block-10 (i.e., before final block)
            # 4. Split: [CLS at tap point], [patch tokens at tap point]
            cls_at_tap = seq[:, 0:1, :]              # [B, 1, 768]
            patches_at_tap = seq[:, 1:, :]            # [B, 196, 768]
            # 5. Reshape patches to spatial 14×14 to match the LAD pipeline's expectations
            z_4d = patches_at_tap.transpose(1, 2).reshape(n, 768, 14, 14).contiguous()
            return z_4d, cls_at_tap

        def h_2d(z: torch.Tensor, side: Optional[torch.Tensor] = None) -> torch.Tensor:
            if vit_cls_mode == "preserve" and side is None:
                raise RuntimeError(
                    "vit_b_16 h_2d (preserve mode) requires the CLS state from g(...). "
                    "Call: z, cls = g(x); logits = h_2d(z_or_recon, cls). "
                    "Pass vit_cls_mode='mean' to derive CLS from the patch mean instead."
                )
            n, c, h, w = z.shape
            if c != 768:
                raise RuntimeError(f"vit_b_16 h_2d expected channel dim 768, got {z.shape}")
            # Sobol' importance feeds [N, C, 1, 1] (a GAP-no-op for CNN heads) — for ViT we
            # broadcast each pooled vector across the 14×14 grid (every patch token = same
            # pooled feature). Standard path uses [N, C, 14, 14] unchanged.
            if (h, w) == (1, 1):
                z = z.expand(-1, -1, 14, 14)
            elif (h, w) != (14, 14):
                raise RuntimeError(f"vit_b_16 h_2d expected (1,1) or (14,14) spatial, got {(h, w)}")
            # Reshape spatial back to token sequence
            patches = z.reshape(n, c, 14 * 14).transpose(1, 2)  # [N, 196, 768]
            if vit_cls_mode == "mean":
                # Interpretation (b): CLS = spatial mean of (possibly reconstructed) patches.
                cls_used = patches.mean(dim=1, keepdim=True)  # [N, 1, 768]
            else:
                # Interpretation (a): preserve the CLS computed from the original patches.
                cls_used = side
                if cls_used.shape[0] != n:
                    cls_used = cls_used.expand(n, -1, -1)
            seq = torch.cat([cls_used, patches], dim=1)  # [N, 197, 768]
            seq = last_block(seq)
            seq = final_ln(seq)
            return heads(seq[:, 0])  # [N, 1000]

        return g, h_2d

    raise ValueError(f"Unknown backbone {backbone!r}")


def feature_dims(backbone: str) -> Tuple[int, int, int]:
    """Return (C, h, w) for the encoder output of the given backbone at 224x224 input."""
    table = {
        "resnet18": (512, 7, 7),
        "resnet34": (512, 7, 7),
        "resnet50": (2048, 7, 7),
        "resnet50_places365": (2048, 7, 7),
        "mobilenet_v2": (1280, 7, 7),
        "convnext_tiny": (768, 7, 7),
        "convnext_small": (768, 7, 7),
        "convnext_small_ham10000": (768, 7, 7),
        "convnext_small_ham10000_448": (768, 14, 14),  # 448px fine-tune
        "densenet121": (1024, 7, 7),
        "vit_b_16": (768, 14, 14),
    }
    if backbone.lower() not in table:
        raise ValueError(f"feature_dims unknown for {backbone!r}")
    return table[backbone.lower()]


__all__ = [
    "load_backbone",
    "make_g_and_h2d",
    "feature_dims",
    "build_classifier",
    "load_classifier_checkpoint",
]
