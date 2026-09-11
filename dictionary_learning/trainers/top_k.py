"""Checkpoint-compatible TopK sparse autoencoder.

This is the small inference-only portion of ``saprmarks/dictionary_learning``
needed to load the checkpoints used in the study.  Training infrastructure is
intentionally omitted.  The upstream project is MIT licensed; attribution and
the license text are included at the repository root.
"""

from __future__ import annotations

from os import PathLike
from typing import Optional, Union

import torch
from torch import Tensor, nn


class AutoEncoderTopK(nn.Module):
    """Top-k SAE with the state-dict layout used by the study checkpoints."""

    def __init__(self, activation_dim: int, dict_size: int, k: int) -> None:
        super().__init__()
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int32))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(
        self,
        x: Tensor,
        return_topk: bool = False,
        use_threshold: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Encode activations using the stored threshold or exact top-k rule."""
        post_relu = torch.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded = post_relu * (post_relu > self.threshold)
            if return_topk:
                selected = post_relu.topk(int(self.k.item()), sorted=False, dim=-1)
                return encoded, selected.values, selected.indices, post_relu
            return encoded

        selected = post_relu.topk(int(self.k.item()), sorted=False, dim=-1)
        encoded = torch.zeros_like(post_relu).scatter_(
            dim=-1,
            index=selected.indices,
            src=selected.values,
        )
        if return_topk:
            return encoded, selected.values, selected.indices, post_relu
        return encoded

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features) + self.b_dec

    def forward(
        self,
        x: Tensor,
        output_features: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        if output_features:
            return reconstruction, features
        return reconstruction

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, PathLike[str]],
        k: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "AutoEncoderTopK":
        """Load a weights-only checkpoint without unpickling arbitrary objects."""
        state = torch.load(path, map_location=device or "cpu", weights_only=True)
        dict_size, activation_dim = state["encoder.weight"].shape
        checkpoint_k = int(state["k"].item())
        if k is not None and k != checkpoint_k:
            raise ValueError(f"requested k={k}, checkpoint has k={checkpoint_k}")

        autoencoder = cls(activation_dim, dict_size, checkpoint_k if k is None else k)
        autoencoder.load_state_dict(state)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
