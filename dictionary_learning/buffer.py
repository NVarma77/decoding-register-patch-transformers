"""Historical vision-token selection used by the checkpoint provenance audit.

Only the selector required to execute the audit is shipped here. The broader
training activation buffer belongs to the source training repository and is not
needed to load or evaluate the released checkpoints.

The ``registers_only`` behavior is intentionally preserved exactly as it ran
during training: it selects the final ``num_register_tokens`` positions. For
Hugging Face DINOv2-with-registers those positions are terminal patches, not the
true registers at positions ``[1:5)``. Correcting this function would erase the
provenance fact that the audit is designed to demonstrate.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def _get_token_subset(cfg: Any) -> str:
    return str(getattr(cfg, "token_subset", "all"))


def _get_num_register_tokens(cfg: Any) -> int:
    return int(getattr(cfg, "num_register_tokens", 4))


def _get_outlier_threshold(cfg: Any) -> float | None:
    return getattr(cfg, "outlier_threshold", None)


def _select_vision_tokens(
    hidden_states: Tensor,
    cfg: Any,
    training: bool = True,
) -> Tensor:
    """Execute the vision-token selector used for the released training runs."""
    if hidden_states.ndim != 3:
        return hidden_states

    has_registers = "with-registers" in str(cfg.model_name).lower()
    token_subset = _get_token_subset(cfg)
    num_register_tokens = _get_num_register_tokens(cfg)

    if not training:
        if has_registers:
            return hidden_states[:, 1:-num_register_tokens, :]
        return hidden_states[:, 1:, :]

    if token_subset == "all":
        return hidden_states[:, 1:, :]

    if token_subset == "registers_only":
        if not has_registers:
            raise ValueError(
                "token_subset='registers_only' requires a with-registers model"
            )
        return hidden_states[:, -num_register_tokens:, :]

    if token_subset == "outlier_patches":
        if has_registers:
            return hidden_states[:, 1:-num_register_tokens, :]
        return hidden_states[:, 1:, :]

    raise ValueError(f"Unknown token_subset: {token_subset}")


__all__ = ["_select_vision_tokens"]
