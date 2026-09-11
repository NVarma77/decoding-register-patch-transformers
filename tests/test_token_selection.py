from types import SimpleNamespace

import pytest
import torch

from dictionary_learning.buffer import _select_vision_tokens


def _positions() -> torch.Tensor:
    return torch.arange(261, dtype=torch.float32).reshape(1, 261, 1)


def _cfg(token_subset: str, model_name: str = "facebook/dinov2-with-registers-small") -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        token_subset=token_subset,
        num_register_tokens=4,
    )


def test_historical_registers_only_selector_targets_terminal_patches() -> None:
    selected = _select_vision_tokens(_positions(), _cfg("registers_only"), training=True)
    assert selected.reshape(-1).tolist() == [257.0, 258.0, 259.0, 260.0]
    assert selected.reshape(-1).tolist() != [1.0, 2.0, 3.0, 4.0]


def test_all_training_positions_drop_only_cls() -> None:
    selected = _select_vision_tokens(_positions(), _cfg("all"), training=True)
    assert selected.shape == (1, 260, 1)
    assert selected[0, 0, 0].item() == 1.0
    assert selected[0, -1, 0].item() == 260.0


def test_registers_only_rejects_model_without_registers() -> None:
    with pytest.raises(ValueError, match="with-registers"):
        _select_vision_tokens(
            _positions(),
            _cfg("registers_only", model_name="facebook/dinov2-small"),
            training=True,
        )


def test_unknown_subset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown token_subset"):
        _select_vision_tokens(_positions(), _cfg("not-a-subset"), training=True)
