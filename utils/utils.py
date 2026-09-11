"""Focused model-loading helpers used by the released experiments."""

from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModel


def resolve_attr(obj: Any, attr_path: str) -> Any:
    """Resolve dotted attributes with optional integer indices.

    Example: ``encoder.layer[8].attention``.
    """
    for part in re.split(r"\.(?![^\[]*\])", attr_path):
        match = re.fullmatch(r"([A-Za-z0-9_]+)(?:\[(\d+)\])?", part)
        if match is None:
            raise ValueError(f"invalid attribute path component: {part!r}")
        obj = getattr(obj, match.group(1))
        if match.group(2) is not None:
            obj = obj[int(match.group(2))]
    return obj


def load_model(
    model_name: str,
    cfg: Any,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0",
    **_kwargs: Any,
) -> tuple[torch.nn.Module, None, AutoImageProcessor]:
    """Load a DINOv2 vision model and its image processor from Hugging Face."""
    if "dinov2" not in model_name.lower():
        raise ValueError(
            "the released vision experiments support DINOv2 models only; "
            "language-model scripts load their models directly"
        )

    model_path = getattr(cfg, "model_path", None) or model_name
    model = AutoModel.from_pretrained(model_path)
    model.to(device=device, dtype=dtype)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(model_path, do_convert_rgb=True)
    return model, None, processor
