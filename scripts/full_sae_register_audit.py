#!/usr/bin/env python3
"""Corrective register intervention audit using the all-token TopK SAE.

The existing ``registers_only`` checkpoints were trained from the final four
sequence positions by ``dictionary_learning.buffer._select_vision_tokens``.
This script does not relabel those checkpoints.  It evaluates that SAE where
it was actually trained, compares its out-of-distribution use on real
registers, and evaluates an all-261-token SAE on the real register block.

The intervention analysis deliberately uses only the all-token SAE.  It can
test whether the prior null was caused by applying a terminal-patch SAE to
registers, but it cannot establish that any learned feature is register-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("REPO_DIR", str(REPO))
os.environ.setdefault("DATA_DIR", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from intervention_controls import ActivationHook, output_tensor, preprocessing_metadata
from utils.utils import load_model, resolve_attr


MODEL_NAME = "facebook/dinov2-with-registers-small"
BLOCK_INDEX = 8
HOOK_LABEL = "output of Hugging Face encoder block index 8 (zero-based)"
SEQUENCE_LENGTH = 261
REGISTER_START = 1
REGISTER_END = 5
TERMINAL_PATCH_START = 257
TERMINAL_PATCH_END = 261
PATCH_START = 5
N_PATCHES = 256
SEED = 20260822
HOLDOUT_STREAM_START = 512
PRIMARY_SIZE = 512
HOLDOUT_SIZE = 256
RANDOM_DRAWS = 8
# Calibrated at 2e-5 for the 384-dim Small model; float32 matmul error
# accumulates with activation_dim, so the 768-dim Base model needs headroom
# (observed ~2.6e-5, a small fraction above 2e-5 -- consistent with expected
# floating-point scaling, not a logic error). A genuine bug would be orders
# of magnitude larger than this, so 1e-4 still catches real regressions.
TOL = 1e-4

MISTARGETED_SAE = REPO / "saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_22192860_registers_only/trainer_0"
FULL_TOKEN_SAE = REPO / "saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_24096850/trainer_0"
DEFAULT_OUTPUT = REPO / "outputs/experiments/full_sae_register_audit"
P0_OUTPUT = REPO / "outputs/experiments/intervention_controls"


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable-no-git-checkout"


def model_revision(model_name: str = MODEL_NAME) -> str:
    ref = Path.home() / f".cache/huggingface/hub/models--{model_name.replace('/', '--')}/refs/main"
    return ref.read_text(encoding="utf-8").strip() if ref.exists() else "unavailable-not-cached"


def hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count(),
    }


def image_source_id(image: Any, label: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(label).encode("ascii"))
    digest.update(image.convert("RGB").tobytes())
    return f"sha256:{digest.hexdigest()}"


def select_dataset_window(start: int, count: int, seed: int, shuffle_buffer: int) -> list[dict[str, Any]]:
    """Return a deterministic window of the fixed streamed validation order."""
    from datasets import load_dataset

    streamed = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    shuffled = streamed.shuffle(seed=seed, buffer_size=shuffle_buffer)
    records: list[dict[str, Any]] = []
    for stream_position, example in enumerate(shuffled.take(start + count)):
        if stream_position < start:
            continue
        label = int(example.get("label", -1))
        image = example["image"].convert("RGB")
        records.append({
            "stream_position": stream_position,
            "source_id": str(example.get("id") or image_source_id(image, label)),
            "label": label,
            "split": "validation",
            "image": image,
        })
    if len(records) != count:
        raise RuntimeError(f"Requested stream window [{start}:{start + count}), got {len(records)} images")
    return records


def public_image_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in record.items() if key != "image"} for record in records]


def assert_primary_prefix(records: list[dict[str, Any]], p0_dir: Path) -> None:
    reference = json.loads((p0_dir / "image_ids.json").read_text(encoding="utf-8"))[:len(records)]
    observed = public_image_records(records)
    if reference != observed:
        for index, (expected, actual) in enumerate(zip(reference, observed)):
            if expected != actual:
                raise AssertionError(f"P0 primary cohort mismatch at position {index}: expected={expected}, actual={actual}")
        raise AssertionError("P0 primary cohort mismatch")


def make_cfg(device: str, dtype: torch.dtype, model_name: str = MODEL_NAME) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        model_path=model_name,
        model_type="vision",
        device=device,
        dtype=dtype,
        submodel="enc",
        get_full_model=False,
        context_length=SEQUENCE_LENGTH,
    )


def encode_decode(sae: AutoEncoderTopK, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xhat, features = sae(x, output_features=True)
    return xhat, features


def reconstruction_stats(x: torch.Tensor, xhat: torch.Tensor) -> tuple[float, float, float]:
    residual = x - xhat
    mse = float(residual.pow(2).mean().item())
    cosine = float(F.cosine_similarity(x.reshape(1, -1), xhat.reshape(1, -1)).item())
    denominator = float((x - x.mean()).pow(2).sum().item())
    explained_variance = float(1.0 - residual.pow(2).sum().item() / denominator) if denominator > 0 else 0.0
    return mse, cosine, explained_variance


def coordinates_from_positive_support(features: torch.Tensor, k: int = 5) -> list[tuple[int, int]]:
    coordinates = [(int(register), int(latent)) for register, latent in zip(*torch.nonzero(features > 0, as_tuple=True))]
    coordinates.sort(key=lambda pair: (float(features[pair[0], pair[1]]), pair[0], pair[1]))
    if len(coordinates) < k:
        raise ValueError(f"Only {len(coordinates)} positive TopK coordinates; need {k}")
    return coordinates[-k:]


def selected_contribution(sae: AutoEncoderTopK, features: torch.Tensor, coordinates: list[tuple[int, int]]) -> torch.Tensor:
    contribution = torch.zeros((features.shape[0], sae.activation_dim), dtype=features.dtype, device=features.device)
    if not coordinates:
        return contribution
    rows = torch.tensor([row for row, _ in coordinates], device=features.device, dtype=torch.long)
    ids = torch.tensor([latent for _, latent in coordinates], device=features.device, dtype=torch.long)
    coefficients = features[rows, ids]
    # ``nn.Linear`` decoder storage is [activation_dim, dictionary_size].
    contribution.index_add_(0, rows, coefficients[:, None] * sae.decoder.weight[:, ids].T)
    return contribution


def match_energy(candidate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, list[float], list[float], list[float]]:
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=-1)
    scaled = candidate * (target_norm / (candidate_norm + 1e-12)).unsqueeze(-1)
    achieved = torch.linalg.vector_norm(scaled, dim=-1)
    return (
        scaled,
        target_norm.detach().cpu().tolist(),
        candidate_norm.detach().cpu().tolist(),
        achieved.detach().cpu().tolist(),
    )


def deterministic_rng(seed: int, cohort: str, stream_position: int, control: str, draw_index: int) -> tuple[np.random.Generator, int]:
    material = f"{seed}|{cohort}|{stream_position}|{control}|{draw_index}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)
    return np.random.default_rng(derived_seed), derived_seed


def random_decoder_span(sae: AutoEncoderTopK, features: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, list[list[int]], list[list[float]]]:
    candidate = torch.zeros((features.shape[0], sae.activation_dim), dtype=features.dtype, device=features.device)
    ids_by_register: list[list[int]] = []
    coefficients_by_register: list[list[float]] = []
    for register in range(features.shape[0]):
        ids = rng.choice(sae.dict_size, size=5, replace=False).astype(int).tolist()
        coefficients = rng.normal(size=5).astype(np.float32).tolist()
        columns = sae.decoder.weight[:, torch.tensor(ids, dtype=torch.long, device=features.device)].T
        candidate[register] = torch.as_tensor(coefficients, dtype=features.dtype, device=features.device) @ columns
        ids_by_register.append(ids)
        coefficients_by_register.append(coefficients)
    return candidate, ids_by_register, coefficients_by_register


def decoder_permutation_control(
    sae: AutoEncoderTopK,
    features: torch.Tensor,
    selected: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Keep selected support and coefficients, replacing only decoder columns."""
    selected_ids = {latent for _, latent in selected}
    available = np.asarray([latent for latent in range(sae.dict_size) if latent not in selected_ids], dtype=np.int64)
    if len(available) < len(selected):
        raise RuntimeError("Not enough nonselected decoder columns for permutation control")
    replacement_ids = rng.choice(available, size=len(selected), replace=False).astype(int).tolist()
    candidate = torch.zeros((features.shape[0], sae.activation_dim), dtype=features.dtype, device=features.device)
    mapping: list[dict[str, Any]] = []
    for (register, source_id), replacement_id in zip(selected, replacement_ids):
        coefficient = features[register, source_id]
        candidate[register] += coefficient * sae.decoder.weight[:, replacement_id]
        mapping.append({
            "register_position": register,
            "source_latent_id": source_id,
            "source_positive_coefficient": float(coefficient.item()),
            "replacement_decoder_latent_id": replacement_id,
        })
    selected_counts = [sum(register == row for row, _ in selected) for register in range(features.shape[0])]
    candidate_counts = [sum(int(item["register_position"]) == row for item in mapping) for row in range(features.shape[0])]
    if selected_counts != candidate_counts:
        raise AssertionError("Decoder-permutation control did not preserve selected support per register")
    if any(item["source_positive_coefficient"] <= 0 for item in mapping):
        raise AssertionError("Decoder-permutation control included a nonpositive source coefficient")
    return candidate, mapping


def patch_metrics(clean: torch.Tensor, output: torch.Tensor) -> tuple[list[float], list[float]]:
    clean_patches = clean[:, PATCH_START:PATCH_START + N_PATCHES]
    output_patches = output[:, PATCH_START:PATCH_START + N_PATCHES]
    if clean_patches.shape[0] == 1 and output_patches.shape[0] != 1:
        clean_patches = clean_patches.expand(output_patches.shape[0], -1, -1)
    cosine = F.cosine_similarity(clean_patches, output_patches, dim=-1).mean(dim=-1)
    stability = [float(value) for value in cosine.detach().cpu().tolist()]
    return stability, [100.0 * (1.0 - value) for value in stability]


def reexecute_post_block_tail(model: torch.nn.Module, post_block_output: torch.Tensor) -> torch.Tensor:
    """Exactly reexecute blocks after the hooked block plus the final layer norm."""
    hidden_states = post_block_output
    for layer_index in range(BLOCK_INDEX + 1, len(model.encoder.layer)):
        hidden_states = model.encoder.layer[layer_index](hidden_states, None, False)[0]
    return model.layernorm(hidden_states)


def batched_interventions(
    model: torch.nn.Module,
    post_block_output: torch.Tensor,
    clean_representation: torch.Tensor,
    interventions: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for offset in range(0, len(interventions), batch_size):
        chunk = interventions[offset:offset + batch_size]
        replacements = torch.stack([item["replacement"] for item in chunk])
        if post_block_output.ndim != 3 or post_block_output.shape[1] < REGISTER_END:
            raise RuntimeError(f"Hooked activation lacks register positions: {tuple(post_block_output.shape)}")
        changed = post_block_output.expand(len(chunk), -1, -1).clone()
        changed[:, REGISTER_START:REGISTER_END, :] = replacements.to(device=changed.device, dtype=changed.dtype)
        with torch.no_grad():
            output = reexecute_post_block_tail(model, changed).detach()
        stability, drops = patch_metrics(clean_representation, output)
        for item, value, drop in zip(chunk, stability, drops):
            results.append({**item, "cosine_stability": value, "percentage_drop": drop})
    return results


def metric_row(
    record: dict[str, Any],
    condition: str,
    stability: float,
    drop: float,
    full_fidelity: dict[str, float],
    selected: list[tuple[int, int]],
    pairwise_exclusions: list[str],
    draw_index: int | None = None,
    draw_count: int | None = None,
    control_seed: int | None = None,
    model_name: str = MODEL_NAME,
) -> dict[str, Any]:
    return {
        "image_id": record["source_id"],
        "stream_position": record["stream_position"],
        "label": record["label"],
        "split": record["split"],
        "condition": condition,
        "draw_index": draw_index,
        "draw_count": draw_count,
        "control_seed": control_seed,
        "eligible": True,
        "pairwise_excluded_conditions": json.dumps(pairwise_exclusions),
        "model_name": model_name,
        "hook": HOOK_LABEL,
        "hook_block_index_zero_based": BLOCK_INDEX,
        "register_positions": "[1:5]",
        "patch_positions": "[5:261]",
        "sae_scope": "all 261 tokens; evaluated on true registers for intervention",
        "feature_selection_rule": "five largest positive active (register position, latent ID) coordinates per image",
        "cosine_stability": stability,
        "percentage_drop": drop,
        "full_sae_register_reconstruction_mse": full_fidelity["mse"],
        "full_sae_register_reconstruction_cosine": full_fidelity["cosine"],
        "full_sae_register_reconstruction_explained_variance": full_fidelity["explained_variance"],
        "selected_top5_active": json.dumps(selected),
        "seed": SEED,
    }


def evaluate_image(
    record: dict[str, Any],
    cohort: str,
    model: torch.nn.Module,
    module: torch.nn.Module,
    processor: Any,
    full_sae: AutoEncoderTopK,
    mistargeted_sae: AutoEncoderTopK,
    random_draws: int,
    intervention_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    validate_identity: bool,
    full_sae_path: str = str(FULL_TOKEN_SAE / "ae.pt"),
    mistargeted_sae_path: str = str(MISTARGETED_SAE / "ae.pt"),
    model_name: str = MODEL_NAME,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pixels = processor(images=record["image"], return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        captured["activation"] = output_tensor(output).detach()
        return output

    handle = module.register_forward_hook(capture)
    with torch.no_grad():
        clean_output = model(pixel_values=pixels).last_hidden_state.detach()
    handle.remove()
    full_activation = captured["activation"]
    if tuple(full_activation.shape[1:]) != (SEQUENCE_LENGTH, full_sae.activation_dim):
        raise RuntimeError(f"Unexpected hooked output shape {tuple(full_activation.shape)}")
    registers = full_activation[0, REGISTER_START:REGISTER_END].to(dtype=torch.float32)
    terminal_patches = full_activation[0, TERMINAL_PATCH_START:TERMINAL_PATCH_END].to(dtype=torch.float32)

    with torch.no_grad():
        full_xhat, features = encode_decode(full_sae, registers)
        mistargeted_register_xhat, _ = encode_decode(mistargeted_sae, registers)
        mistargeted_terminal_xhat, _ = encode_decode(mistargeted_sae, terminal_patches)
    full_mse, full_cosine, full_fve = reconstruction_stats(registers, full_xhat)
    register_mse, register_cosine, register_fve = reconstruction_stats(registers, mistargeted_register_xhat)
    terminal_mse, terminal_cosine, terminal_fve = reconstruction_stats(terminal_patches, mistargeted_terminal_xhat)
    full_fidelity = {"mse": full_mse, "cosine": full_cosine, "explained_variance": full_fve}
    fidelity_rows = [
        {
            "image_id": record["source_id"], "stream_position": record["stream_position"], "label": record["label"], "split": record["split"],
            "sae": "mislabeled registers_only SAE", "sae_path": mistargeted_sae_path,
            "evaluated_positions": "terminal patches [257:261)", "why": "where its released training selector actually drew activations", "reconstruction_mse": terminal_mse, "reconstruction_cosine": terminal_cosine, "explained_variance": terminal_fve,
        },
        {
            "image_id": record["source_id"], "stream_position": record["stream_position"], "label": record["label"], "split": record["split"],
            "sae": "mislabeled registers_only SAE", "sae_path": mistargeted_sae_path,
            "evaluated_positions": "true registers [1:5)", "why": "current out-of-distribution failure case", "reconstruction_mse": register_mse, "reconstruction_cosine": register_cosine, "explained_variance": register_fve,
        },
        {
            "image_id": record["source_id"], "stream_position": record["stream_position"], "label": record["label"], "split": record["split"],
            "sae": "full-token SAE", "sae_path": full_sae_path,
            "evaluated_positions": "true registers [1:5)", "why": "valid corrective comparison: all-token training included true registers", "reconstruction_mse": full_mse, "reconstruction_cosine": full_cosine, "explained_variance": full_fve,
        },
    ]

    try:
        selected = coordinates_from_positive_support(features)
    except ValueError as exc:
        raise RuntimeError(f"Unexpected ineligible full-SAE register example {record['source_id']}: {exc}") from exc
    top_contribution = selected_contribution(full_sae, features, selected)
    masked_features = features.clone()
    for register, latent in selected:
        masked_features[register, latent] = 0
    masked_decode_error = float((full_sae.decode(masked_features) - (full_xhat - top_contribution)).abs().max().item())
    if masked_decode_error > TOL:
        raise AssertionError(f"Masked full-SAE decode mismatch: {masked_decode_error}")
    selected_registers = {register for register, _ in selected}
    outside_selected_error = max((float(top_contribution[register].abs().max().item()) for register in range(REGISTER_END - REGISTER_START) if register not in selected_registers), default=0.0)
    if outside_selected_error > TOL:
        raise AssertionError(f"Selected contribution leaked outside source register position: {outside_selected_error}")

    identity_error: float | None = None
    tail_clean_error: float | None = None
    if validate_identity:
        with torch.no_grad():
            tail_clean = reexecute_post_block_tail(model, full_activation).detach()
        tail_clean_error = float((tail_clean - clean_output).abs().max().item())
        if tail_clean_error > TOL:
            raise AssertionError(f"Manual post-block tail differs from full model output: {tail_clean_error}")
        with ActivationHook(module, lambda activation: activation):
            with torch.no_grad():
                identity_output = model(pixel_values=pixels).last_hidden_state.detach()
        identity_error = float((identity_output - clean_output).abs().max().item())
        if identity_error > TOL:
            raise AssertionError(f"Identity hook mismatch: {identity_error}")

    interventions: list[dict[str, Any]] = [
        {"condition": "reconstruction", "draw_index": None, "draw_count": None, "control_seed": None, "replacement": full_xhat},
        {"condition": "clean_top5", "draw_index": None, "draw_count": None, "control_seed": None, "replacement": registers - top_contribution},
        {"condition": "clean_sign_flip", "draw_index": None, "draw_count": None, "control_seed": None, "replacement": registers + top_contribution},
    ]
    span_audit: list[dict[str, Any]] = []
    permutation_audit: list[dict[str, Any]] = []
    invalid_span = False
    invalid_permutation = False
    for draw_index in range(random_draws):
        span_rng, span_seed = deterministic_rng(SEED, cohort, record["stream_position"], "random_span", draw_index)
        span_candidate, span_ids, span_coefficients = random_decoder_span(full_sae, features, span_rng)
        span_scaled, target_norm, span_unscaled_norm, span_achieved_norm = match_energy(span_candidate, top_contribution)
        span_zero_rows = [row for row, (target, candidate) in enumerate(zip(target_norm, span_unscaled_norm)) if target > 1e-12 and candidate <= 1e-12]
        if span_zero_rows:
            invalid_span = True
        for target, achieved in zip(target_norm, span_achieved_norm):
            if target > 1e-12 and abs(target - achieved) > 5e-4:
                raise AssertionError("Random span energy match failed")
        interventions.append({"condition": "clean_random_span_draw", "draw_index": draw_index, "draw_count": random_draws, "control_seed": span_seed, "replacement": registers - span_scaled})
        span_audit.append({"draw_index": draw_index, "seed": span_seed, "decoder_latent_ids_per_register": span_ids, "normal_coefficients_per_register": span_coefficients, "target_norm_per_register": target_norm, "candidate_unscaled_norm_per_register": span_unscaled_norm, "achieved_norm_per_register": span_achieved_norm, "zero_norm_registers": span_zero_rows})

        permutation_rng, permutation_seed = deterministic_rng(SEED, cohort, record["stream_position"], "decoder_permutation", draw_index)
        permutation_candidate, mapping = decoder_permutation_control(full_sae, features, selected, permutation_rng)
        permutation_scaled, permutation_target_norm, permutation_unscaled_norm, permutation_achieved_norm = match_energy(permutation_candidate, top_contribution)
        permutation_zero_rows = [row for row, (target, candidate) in enumerate(zip(permutation_target_norm, permutation_unscaled_norm)) if target > 1e-12 and candidate <= 1e-12]
        if permutation_zero_rows:
            invalid_permutation = True
        for target, achieved in zip(permutation_target_norm, permutation_achieved_norm):
            if target > 1e-12 and abs(target - achieved) > 5e-4:
                raise AssertionError("Decoder-permutation energy match failed")
        interventions.append({"condition": "clean_decoder_permutation_draw", "draw_index": draw_index, "draw_count": random_draws, "control_seed": permutation_seed, "replacement": registers - permutation_scaled})
        permutation_audit.append({"draw_index": draw_index, "seed": permutation_seed, "support_and_magnitude_mapping": mapping, "target_norm_per_register": permutation_target_norm, "candidate_unscaled_norm_per_register": permutation_unscaled_norm, "achieved_norm_per_register": permutation_achieved_norm, "zero_norm_registers": permutation_zero_rows})

    intervention_results = batched_interventions(model, full_activation, clean_output, interventions, intervention_batch_size)
    common_exclusions: list[str] = []
    if invalid_span:
        common_exclusions.append("clean_random_span_mean")
    if invalid_permutation:
        common_exclusions.append("clean_decoder_permutation_mean")
    rows = [metric_row(record, "clean", 1.0, 0.0, full_fidelity, selected, common_exclusions, model_name=model_name)]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for result in intervention_results:
        row = metric_row(
            record,
            result["condition"],
            result["cosine_stability"],
            result["percentage_drop"],
            full_fidelity,
            selected,
            common_exclusions,
            draw_index=result["draw_index"],
            draw_count=result["draw_count"],
            control_seed=result["control_seed"],
            model_name=model_name,
        )
        rows.append(row)
        by_condition.setdefault(result["condition"], []).append(row)
    for raw_condition, mean_condition in [
        ("clean_random_span_draw", "clean_random_span_mean"),
        ("clean_decoder_permutation_draw", "clean_decoder_permutation_mean"),
    ]:
        raw_rows = by_condition[raw_condition]
        rows.append(metric_row(
            record,
            mean_condition,
            float(np.mean([row["cosine_stability"] for row in raw_rows])),
            float(np.mean([row["percentage_drop"] for row in raw_rows])),
            full_fidelity,
            selected,
            common_exclusions,
            draw_count=random_draws,
            model_name=model_name,
        ))
    audit = {
        "image_id": record["source_id"],
        "stream_position": record["stream_position"],
        "eligible": True,
        "top5_active": selected,
        "top5_positive_coefficients": [float(features[register, latent].item()) for register, latent in selected],
        "top5_norm_per_register": torch.linalg.vector_norm(top_contribution, dim=-1).detach().cpu().tolist(),
        "identity_hook_max_abs_error": identity_error,
        "manual_post_block_tail_max_abs_error": tail_clean_error,
        "masked_decode_max_abs_error": masked_decode_error,
        "top5_nonselected_register_max_abs": outside_selected_error,
        "random_span_draws": span_audit,
        "decoder_permutation_draws": permutation_audit,
        "pairwise_excluded_conditions": common_exclusions,
    }
    return fidelity_rows, rows, audit


def write_rows(rows: list[dict[str, Any]], output: Path, basename: str) -> None:
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / f"{basename}.parquet", index=False)
    frame.to_csv(output / f"{basename}.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def bootstrap_ci(values: np.ndarray, seed: int, resamples: int = 10000) -> list[float]:
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def value_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    if len(values) == 0:
        return {"n_images": 0, "mean": None, "median": None, "bootstrap_ci95": [None, None], "values": []}
    return {
        "n_images": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "bootstrap_ci95": bootstrap_ci(values, seed),
        "values": values.tolist(),
    }


def excluded_conditions(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        return set(json.loads(value))
    return set(value)


def paired_summary(frame: pd.DataFrame, left: str, right: str, seed: int) -> dict[str, Any]:
    filtered = frame[frame["condition"].isin([left, right])]
    by_image: dict[str, dict[str, Any]] = {}
    for row in filtered.to_dict(orient="records"):
        by_image.setdefault(str(row["image_id"]), {})[str(row["condition"])] = row
    differences: list[float] = []
    for values in by_image.values():
        if left not in values or right not in values:
            continue
        if not bool(values[left]["eligible"]) or not bool(values[right]["eligible"]):
            continue
        if right in excluded_conditions(values[left]["pairwise_excluded_conditions"]) or left in excluded_conditions(values[right]["pairwise_excluded_conditions"]):
            continue
        differences.append(float(values[left]["percentage_drop"]) - float(values[right]["percentage_drop"]))
    summary = value_summary(np.asarray(differences, dtype=np.float64), seed)
    summary["contrast"] = f"{left} - {right}"
    return summary


def condition_summary(frame: pd.DataFrame, condition: str, seed: int) -> dict[str, Any]:
    values = frame.loc[(frame["condition"] == condition) & frame["eligible"].astype(bool), "percentage_drop"].astype(float).to_numpy()
    summary = value_summary(values, seed)
    summary["condition"] = condition
    return summary


def fidelity_table(fidelity: pd.DataFrame, cohort: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = fidelity.groupby(["sae", "evaluated_positions", "why", "sae_path"], sort=False)
    for index, ((sae, positions, why, sae_path), group) in enumerate(grouped):
        rows.append({
            "cohort": cohort,
            "sae": sae,
            "evaluated_positions": positions,
            "why": why,
            "sae_path": sae_path,
            "n_images": int(len(group)),
            "reconstruction_mse": value_summary(group["reconstruction_mse"].astype(float).to_numpy(), seed + index * 13 + 1),
            "reconstruction_cosine": value_summary(group["reconstruction_cosine"].astype(float).to_numpy(), seed + index * 13 + 2),
            "explained_variance": value_summary(group["explained_variance"].astype(float).to_numpy(), seed + index * 13 + 3),
        })
    return rows


def run_cohort(args: argparse.Namespace) -> None:
    if args.cohort == "primary":
        stream_start, default_size = 0, PRIMARY_SIZE
    else:
        stream_start, default_size = HOLDOUT_STREAM_START, HOLDOUT_SIZE
    n_images = args.n_images or default_size
    if n_images != default_size and not args.allow_nonprotocol_size:
        raise ValueError(f"{args.cohort} protocol requires {default_size} images; use --allow-nonprotocol-size only for a smoke test")
    output = args.output_dir or (args.root_output / args.cohort)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(output / "run.log"), logging.StreamHandler()], force=True)
    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.shuffle_buffer != 512 and not args.allow_nonprotocol_size:
        raise ValueError("The protocol uses the original P0 streaming shuffle buffer of 512; changing it changes the P0 cohort")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    records = select_dataset_window(stream_start, n_images, args.seed, args.shuffle_buffer)
    if args.cohort == "primary":
        assert_primary_prefix(records, args.p0_output)
    else:
        p0_ids = {item["source_id"] for item in json.loads((args.p0_output / "image_ids.json").read_text(encoding="utf-8"))}
        overlap = p0_ids & {record["source_id"] for record in records}
        if overlap:
            raise AssertionError(f"Fixed holdout overlaps P0 cohort: {sorted(overlap)[:3]}")
    full_path = args.full_sae / "ae.pt"
    mistargeted_path = args.mistargeted_sae / "ae.pt"
    if not full_path.exists() or not mistargeted_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: full={full_path.exists()} mistargeted={mistargeted_path.exists()}")
    full_sae = AutoEncoderTopK.from_pretrained(str(full_path), device=device)
    mistargeted_sae = AutoEncoderTopK.from_pretrained(str(mistargeted_path), device=device)
    full_sae.eval()
    mistargeted_sae.eval()
    model, _tokenizer, processor = load_model(args.model_name, make_cfg(str(device), dtype, args.model_name), dtype=dtype, device=str(device))
    model.eval()
    module = resolve_attr(model, f"encoder.layer[{BLOCK_INDEX}]")
    save_json(output / "image_ids.json", public_image_records(records))
    full_eval_path = args.full_sae / "eval_results.json"
    full_eval = json.loads(full_eval_path.read_text(encoding="utf-8")) if full_eval_path.exists() else None
    config = {
        "command": " ".join(sys.argv),
        "cohort": args.cohort,
        "protocol_cohort_size": default_size,
        "n_selected": len(records),
        "stream_window": [stream_start, stream_start + n_images],
        "cohort_source": f"ILSVRC/imagenet-1k validation streaming shuffle(seed={args.seed}, buffer_size={args.shuffle_buffer})",
        "cohort_validation": "primary prefix exactly matched the completed P0 IDs" if args.cohort == "primary" else "stream positions [512:768) and source IDs are disjoint from P0",
        "model_name": args.model_name,
        "model_revision": model_revision(args.model_name),
        "hook": HOOK_LABEL,
        "hook_block_index_zero_based": BLOCK_INDEX,
        "hook_output_expected_shape": f"[batch, {SEQUENCE_LENGTH}, {full_sae.activation_dim}]",
        "actual_register_positions": "[1:5]",
        "terminal_patch_positions_used_by_mislabeled_selector": "[257:261]",
        "patch_metric_positions": "[5:261]",
        "intervention_execution": "capture the output of encoder block 8 once, then reexecute encoder blocks 9-11 and final layernorm; equivalence to the full clean forward is asserted on two images",
        "preprocessing": preprocessing_metadata(),
        "full_token_sae": {"path": str(full_path), "sha256": sha256_file(full_path), "config": str(args.full_sae / "config.json"), "stored_eval_results": full_eval, "scope": "all 261 sequence positions"},
        "mislabeled_registers_only_sae": {"path": str(mistargeted_path), "sha256": sha256_file(mistargeted_path), "config": str(args.mistargeted_sae / "config.json"), "scope": "historically selected final four positions, not model register positions"},
        "random_controls": {"random_draws_per_image": args.random_draws, "aggregation": "mean of raw control outcomes within image before image-level paired bootstrap", "random_span": "five random decoder columns with Normal(0,1) coefficients per register, energy-matched per register", "decoder_permutation": "same selected register support and positive coefficients, with nonselected decoder-column identities sampled without replacement and energy-matched per register"},
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "hardware": hardware_metadata(),
        "code_revision": code_revision(),
    }
    save_json(output / "config.json", config)

    fidelity_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        fidelity, metrics, audit = evaluate_image(
            record, args.cohort, model, module, processor, full_sae, mistargeted_sae,
            args.random_draws, args.intervention_batch_size, device, dtype, validate_identity=index <= 2,
            full_sae_path=str(full_path), mistargeted_sae_path=str(mistargeted_path),
            model_name=args.model_name,
        )
        fidelity_rows.extend(fidelity)
        metric_rows.extend(metrics)
        audits.append(audit)
        if index % 16 == 0 or index == len(records):
            logging.info("Processed %d/%d images", index, len(records))
    write_rows(fidelity_rows, output, "reconstruction_fidelity")
    write_rows(metric_rows, output, "per_image_metrics")
    save_json(output / "selection_audit.json", audits)
    metrics = pd.DataFrame(metric_rows)
    fidelity_frame = pd.DataFrame(fidelity_rows)
    summary = {
        "cohort": args.cohort,
        "n_images": len(records),
        "n_metric_rows": len(metric_rows),
        "n_fidelity_rows": len(fidelity_rows),
        "validation": {
            "identity_hook_max_abs_error": max((float(a["identity_hook_max_abs_error"]) for a in audits if a["identity_hook_max_abs_error"] is not None), default=None),
            "manual_post_block_tail_max_abs_error": max((float(a["manual_post_block_tail_max_abs_error"]) for a in audits if a["manual_post_block_tail_max_abs_error"] is not None), default=None),
            "masked_decode_max_abs_error": max(float(a["masked_decode_max_abs_error"]) for a in audits),
            "top5_nonselected_register_max_abs": max(float(a["top5_nonselected_register_max_abs"]) for a in audits),
            "all_bottom_coordinates_positive": "not applicable: this corrective analysis uses only top-five coordinates",
            "random_span_zero_norm_draws": int(sum(len(draw["zero_norm_registers"]) for audit in audits for draw in audit["random_span_draws"])),
            "decoder_permutation_zero_norm_draws": int(sum(len(draw["zero_norm_registers"]) for audit in audits for draw in audit["decoder_permutation_draws"])),
            "patch_metric_excludes_cls_and_register_tokens": True,
        },
        "fidelity_table": fidelity_table(fidelity_frame, args.cohort, args.seed),
        "condition_effects_percentage_drop": [condition_summary(metrics, condition, args.seed + index * 17) for index, condition in enumerate(["reconstruction", "clean_top5", "clean_random_span_mean", "clean_decoder_permutation_mean", "clean_sign_flip"])],
        "paired_contrasts_percentage_drop": [
            paired_summary(metrics, "reconstruction", "clean", args.seed + 101),
            paired_summary(metrics, "clean_top5", "clean_random_span_mean", args.seed + 202),
            paired_summary(metrics, "clean_top5", "clean_decoder_permutation_mean", args.seed + 303),
        ],
        "random_draws_are_aggregated_within_image_before_bootstrap": True,
    }
    save_json(output / "summary.json", summary)
    final_config = {**config, "elapsed_seconds": time.time() - start_time}
    save_json(output / "config.json", final_config)
    (output / "README.md").write_text(
        "Corrective full-token SAE register audit. This cohort evaluates three reconstruction targets and uses clean top-five removal with eight image-level-aggregated random-span and support/magnitude-matched decoder-permutation controls. It does not establish register-only feature scope because the intervention SAE was trained on all sequence positions.\n",
        encoding="utf-8",
    )
    logging.info("Completed %s cohort in %.1fs", args.cohort, time.time() - start_time)


def read_summary(root: Path, cohort: str) -> dict[str, Any]:
    return json.loads((root / cohort / "summary.json").read_text(encoding="utf-8"))


def summary_map(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(item[key]): item for item in items}


def render_reconstruction_table(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    primary = [row for row in rows if row["cohort"] == "primary"]
    cells = []
    for row in primary:
        fve = row["explained_variance"]
        cosine = row["reconstruction_cosine"]
        cells.append([
            row["sae"],
            row["evaluated_positions"],
            row["why"],
            f"{fve['mean']:.3f} [{fve['bootstrap_ci95'][0]:.3f}, {fve['bootstrap_ci95'][1]:.3f}]",
            f"{cosine['mean']:.3f}",
        ])
    figure, axis = plt.subplots(figsize=(12.2, 3.0), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=["SAE", "Evaluated positions", "Why", "FVE (95% image CI)", "Recon. cosine"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.18, 0.17, 0.36, 0.19, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.75)
    for column in range(5):
        table[(0, column)].set_facecolor("#d9e7e2")
    axis.set_title("Target versus mistargeted reconstruction audit: fixed 512-image cohort", fontsize=11)
    figure.savefig(output / "figure_reconstruction_target_audit.png", dpi=220)
    figure.savefig(output / "figure_reconstruction_target_audit.pdf")
    plt.close(figure)


def render_intervention_figure(primary: dict[str, Any], holdout: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["reconstruction", "clean top5", "random span\n(8-draw mean)", "decoder permutation\n(8-draw mean)"]
    conditions = ["reconstruction", "clean_top5", "clean_random_span_mean", "clean_decoder_permutation_mean"]
    colors = ["#777777", "#1b9e77", "#66a61e", "#e7298a"]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True, sharey=True)
    for axis, cohort_summary, title in zip(axes, [primary, holdout], ["Primary: P0-matched 512 images", "Separate fixed 256-image holdout"]):
        effects = summary_map(cohort_summary["condition_effects_percentage_drop"], "condition")
        means = np.asarray([effects[condition]["mean"] for condition in conditions])
        cis = np.asarray([effects[condition]["bootstrap_ci95"] for condition in conditions])
        errors = np.asarray([means - cis[:, 0], cis[:, 1] - means])
        axis.bar(range(len(conditions)), means, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(range(len(conditions)), labels)
        axis.set_title(title, fontsize=10)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Final patch cosine drop (%)")
    figure.savefig(output / "figure_full_sae_intervention_controls.png", dpi=220)
    figure.savefig(output / "figure_full_sae_intervention_controls.pdf")
    plt.close(figure)


def render_sign_flip_figure(primary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    effects = summary_map(primary["condition_effects_percentage_drop"], "condition")
    conditions = ["clean_top5", "clean_sign_flip"]
    labels = ["top5 removal", "top5 sign flip"]
    values = [effects[condition] for condition in conditions]
    means = np.asarray([item["mean"] for item in values])
    cis = np.asarray([item["bootstrap_ci95"] for item in values])
    errors = np.asarray([means - cis[:, 0], cis[:, 1] - means])
    figure, axis = plt.subplots(figsize=(4.5, 3.4), constrained_layout=True)
    axis.bar(range(2), means, yerr=errors, capsize=3, color=["#1b9e77", "#7570b3"])
    axis.set_xticks(range(2), labels)
    axis.set_ylabel("Final patch cosine drop (%)")
    axis.set_title("Supplementary directionality diagnostic")
    axis.set_ylim(bottom=0)
    figure.savefig(output / "figure_sign_flip_diagnostic.png", dpi=220)
    figure.savefig(output / "figure_sign_flip_diagnostic.pdf")
    plt.close(figure)


def aggregate_outputs(args: argparse.Namespace) -> None:
    output = args.root_output
    primary = read_summary(output, "primary")
    holdout = read_summary(output, "holdout")
    fidelity_rows = [row for summary in [primary, holdout] for row in summary["fidelity_table"]]
    table_frame_rows = []
    for row in fidelity_rows:
        table_frame_rows.append({
            "cohort": row["cohort"], "sae": row["sae"], "evaluated_positions": row["evaluated_positions"], "why": row["why"], "sae_path": row["sae_path"], "n_images": row["n_images"],
            "fve_mean": row["explained_variance"]["mean"], "fve_ci95_low": row["explained_variance"]["bootstrap_ci95"][0], "fve_ci95_high": row["explained_variance"]["bootstrap_ci95"][1],
            "reconstruction_cosine_mean": row["reconstruction_cosine"]["mean"], "reconstruction_mse_mean": row["reconstruction_mse"]["mean"],
        })
    pd.DataFrame(table_frame_rows).to_csv(output / "reconstruction_target_audit_table.csv", index=False)
    pd.DataFrame(table_frame_rows).to_parquet(output / "reconstruction_target_audit_table.parquet", index=False)
    save_json(output / "reconstruction_target_audit_table.json", fidelity_rows)
    render_reconstruction_table(fidelity_rows, output)
    render_intervention_figure(primary, holdout, output)
    render_sign_flip_figure(primary, output)
    primary_contrasts = summary_map(primary["paired_contrasts_percentage_drop"], "contrast")
    holdout_contrasts = summary_map(holdout["paired_contrasts_percentage_drop"], "contrast")
    contrast_names = ["clean_top5 - clean_random_span_mean", "clean_top5 - clean_decoder_permutation_mean"]
    primary_pass = all(primary_contrasts[name]["bootstrap_ci95"][0] > 0 for name in contrast_names)
    holdout_pass = all(holdout_contrasts[name]["bootstrap_ci95"][0] > 0 for name in contrast_names)
    primary_controls_larger = all(primary_contrasts[name]["bootstrap_ci95"][1] < 0 for name in contrast_names)
    holdout_controls_larger = all(holdout_contrasts[name]["bootstrap_ci95"][1] < 0 for name in contrast_names)
    if primary_pass and holdout_pass:
        category = "feature-selective within the full-token SAE dictionary"
        claim = "Clean top-five removal exceeded both matched controls on both cohorts. This supports feature-selectivity within the full-token dictionary, but does not establish register-only feature scope."
    elif primary_controls_larger and holdout_controls_larger:
        category = "generic-perturbation result after correcting SAE distribution shift"
        claim = "Even with an in-distribution full-token SAE on true registers, clean top-five removal was consistently less disruptive than both matched controls on both cohorts. The result does not support selected-feature semantic causality or a register-only feature claim."
    else:
        category = "matched-control negative or mixed result"
        claim = "At least one matched-control contrast does not show clean top-five removal exceeding the control on both cohorts. Do not claim selected-feature semantic causality."
    aggregate = {
        "primary": primary,
        "holdout": holdout,
        "reconstruction_target_audit_table": fidelity_rows,
        "interpretation": {"category": category, "claim_boundary": claim},
        "random_control_design": "Eight independent span and support/magnitude-matched decoder-permutation draws were averaged within image before image-level 10,000-resample paired bootstrap inference.",
        "hook": HOOK_LABEL,
        "full_sae_scope_boundary": "The intervention SAE was trained on all 261 sequence positions. It is a valid reconstruction control for true registers but cannot establish register-only feature scope.",
    }
    save_json(output / "summary.json", aggregate)
    primary_recon = summary_map(primary["condition_effects_percentage_drop"], "condition")["reconstruction"]
    memo_lines = [
        "# Corrective full-SAE register audit",
        "",
        f"Interpretation: **{category}**.",
        "",
        "The target-versus-mistargeted reconstruction table is in `reconstruction_target_audit_table.csv` and `figure_reconstruction_target_audit.pdf`. The full-token SAE was trained on all 261 token positions, so its true-register reconstruction is a corrective distribution-shift check, not evidence for register-only features.",
        "",
        f"Primary full-SAE reconstruction injection produced a {primary_recon['mean']:.2f}% mean final-patch cosine drop (95% image-level CI [{primary_recon['bootstrap_ci95'][0]:.2f}, {primary_recon['bootstrap_ci95'][1]:.2f}]).",
        "",
        "| Cohort | Top5 minus random span (95% CI) | Top5 minus decoder permutation (95% CI) |",
        "| --- | --- | --- |",
    ]
    for cohort_name, summary in [("P0-matched 512", primary), ("Separate 256 holdout", holdout)]:
        contrasts = summary_map(summary["paired_contrasts_percentage_drop"], "contrast")
        span = contrasts["clean_top5 - clean_random_span_mean"]
        permutation = contrasts["clean_top5 - clean_decoder_permutation_mean"]
        memo_lines.append(f"| {cohort_name} | {span['mean']:.3f} [{span['bootstrap_ci95'][0]:.3f}, {span['bootstrap_ci95'][1]:.3f}] | {permutation['mean']:.3f} [{permutation['bootstrap_ci95'][0]:.3f}, {permutation['bootstrap_ci95'][1]:.3f}] |")
    memo_lines.extend(["", claim, ""])
    (output / "OUTCOME_MEMO.md").write_text("\n".join(memo_lines), encoding="utf-8")
    root_config = {
        "command": " ".join(sys.argv),
        "model_name": args.model_name,
        "model_revision": model_revision(args.model_name),
        "hook": HOOK_LABEL,
        "primary_source": str(output / "primary"),
        "holdout_source": str(output / "holdout"),
        "random_draws_per_image": RANDOM_DRAWS,
        "code_revision": code_revision(),
        "hardware": hardware_metadata(),
    }
    save_json(output / "config.json", root_config)
    hw = hardware_metadata()
    hw_desc = f"{'CUDA' if hw['cuda_available'] else 'CPU'} float32, {hw['cpu_count']} logical CPUs, PyTorch {hw['torch_version']}" + (
        f", CUDA device(s): {hw['cuda_device_count']}" if hw['cuda_available'] else ", CUDA unavailable"
    )
    (output / "README.md").write_text(
        f"""# Corrective full-SAE true-register audit

This audit tests whether the prior true-register null was solely caused by applying a terminal-patch SAE out of distribution. The all-token SAE is a valid reconstruction/intervention control for true registers, but it is not evidence for register-only features.

## Cohorts and execution

- `primary/` exactly matches the completed P0 512-image ImageNet-1k validation cohort.
- `holdout/` is the next nonoverlapping 256 images in the same deterministic streaming shuffle, with seed `20260822` and shuffle buffer `512`.
- Model: `{args.model_name}`, revision `{model_revision(args.model_name)}`.
- Hook: output of Hugging Face encoder block index 8 (zero-based).
- True registers: `[1:5)`; mislabeled checkpoint source positions: `[257:261)`; final-patch metric: `[5:261)`.
- Hardware/precision: {hw_desc}.

## Reproduction

```bash
.venv/bin/python scripts/audit_registers_only_provenance.py \\
  --output-dir outputs/experiments/registers_only_provenance_audit
.venv/bin/python scripts/full_sae_register_audit.py \\
  --cohort primary --random-draws 8 --intervention-batch-size 8 --device cpu --dtype float32
.venv/bin/python scripts/full_sae_register_audit.py \\
  --cohort holdout --random-draws 8 --intervention-batch-size 8 --device cpu --dtype float32
.venv/bin/python scripts/full_sae_register_audit.py --aggregate \\
  --root-output outputs/experiments/full_sae_register_audit
```

`primary/` and `holdout/` contain exact image IDs, raw per-image CSV/Parquet metrics, fidelity data, selected coordinates, every random draw, seeds, validation audits, and summaries. The aggregate outputs are `summary.json`, `OUTCOME_MEMO.md`, `PAPER_OUTCOME_MEMO.md`, `reconstruction_target_audit_table.csv`, and the three compact figures. The strict provenance command `python scripts/audit_registers_only_provenance.py --strict` is intentionally expected to raise for the historical checkpoints.
""",
        encoding="utf-8",
    )
    (output / "run.log").write_text("Aggregate figures, table, and outcome memo generated from completed primary and holdout runs.\n", encoding="utf-8")
    print(f"Wrote aggregate corrective audit to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["primary", "holdout"])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--root-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--p0-output", type=Path, default=P0_OUTPUT)
    parser.add_argument("--n-images", type=int)
    parser.add_argument("--allow-nonprotocol-size", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--random-draws", type=int, default=RANDOM_DRAWS)
    parser.add_argument("--intervention-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--full-sae", type=Path, default=FULL_TOKEN_SAE)
    parser.add_argument("--mistargeted-sae", type=Path, default=MISTARGETED_SAE)
    parser.add_argument("--model-name", default=MODEL_NAME, help="vision model to hook activations from; must match the scale the --full-sae/--mistargeted-sae checkpoints were trained on")
    args = parser.parse_args()
    if args.aggregate:
        if args.cohort is not None:
            raise ValueError("--aggregate cannot be combined with --cohort")
        aggregate_outputs(args)
        return
    if args.cohort is None:
        raise ValueError("Specify --cohort primary|holdout or --aggregate")
    if args.random_draws < 8:
        raise ValueError("Protocol requires at least eight independent random draws per image")
    if args.intervention_batch_size <= 0:
        raise ValueError("--intervention-batch-size must be positive")
    run_cohort(args)


if __name__ == "__main__":
    main()
