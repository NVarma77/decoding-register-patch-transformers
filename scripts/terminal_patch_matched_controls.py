#!/usr/bin/env python3
"""Matched-control causal ablation at terminal patches [257:261).

F4 (``legacy_protocol_terminal_patch_replay.py``) already replayed the legacy
top-five/bottom-five ablation at terminal patches -- the mislabeled
``registers_only`` checkpoint's own training population, where it is
comparatively well fit (FVE ~0.661) -- but never ran the energy-matched
random-span or support/magnitude-matched decoder-permutation controls there.
F3 (``full_sae_register_audit.py``) ran exactly those controls at true
register positions with a well-fitting full-token SAE and found the sign
reversal (selected top-five removal *less* disruptive than matched controls).

This script asks whether that reversal is specific to register-token
geometry, or whether it is a general property of intervening on a population
an SAE was actually trained on. It reuses the same cohort construction,
selection rule, matched-control generation, and bootstrap machinery as F3
(imported directly from ``full_sae_register_audit`` so the cohort is
byte-identical), pointed at terminal patches with the one SAE that is
actually in-distribution there.

The patch metric here is deliberately narrowed to [5:257) -- excluding CLS,
true registers, and the four intervened terminal positions -- to mirror how
the register experiments exclude the positions being intervened on from
their own metric window. (F4's own metric window, [5:261), included the
intervened terminal positions; this script does not repeat that asymmetry.)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from intervention_controls import ActivationHook, output_tensor, preprocessing_metadata
from utils.utils import load_model, resolve_attr

from full_sae_register_audit import (
    MODEL_NAME, BLOCK_INDEX, HOOK_LABEL, SEQUENCE_LENGTH,
    TERMINAL_PATCH_START, TERMINAL_PATCH_END, SEED, HOLDOUT_STREAM_START,
    PRIMARY_SIZE, HOLDOUT_SIZE, RANDOM_DRAWS, TOL, MISTARGETED_SAE, P0_OUTPUT,
    make_cfg, select_dataset_window, public_image_records, assert_primary_prefix,
    jsonable, save_json, sha256_file, code_revision, model_revision, hardware_metadata,
    encode_decode, reconstruction_stats, coordinates_from_positive_support,
    selected_contribution, match_energy, deterministic_rng, random_decoder_span,
    decoder_permutation_control, reexecute_post_block_tail,
    bootstrap_ci, value_summary, paired_summary, condition_summary,
)

PATCH_METRIC_START = 5
PATCH_METRIC_END = TERMINAL_PATCH_START  # [5:257): excludes CLS, true registers, and the intervened terminal patches
DEFAULT_OUTPUT = REPO / "outputs/experiments/terminal_patch_matched_controls"


def patch_metrics_terminal(clean: torch.Tensor, output: torch.Tensor) -> tuple[list[float], list[float]]:
    clean_patches = clean[:, PATCH_METRIC_START:PATCH_METRIC_END]
    output_patches = output[:, PATCH_METRIC_START:PATCH_METRIC_END]
    if clean_patches.shape[0] == 1 and output_patches.shape[0] != 1:
        clean_patches = clean_patches.expand(output_patches.shape[0], -1, -1)
    cosine = F.cosine_similarity(clean_patches, output_patches, dim=-1).mean(dim=-1)
    stability = [float(value) for value in cosine.detach().cpu().tolist()]
    return stability, [100.0 * (1.0 - value) for value in stability]


def batched_interventions_terminal(
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
        if post_block_output.ndim != 3 or post_block_output.shape[1] < TERMINAL_PATCH_END:
            raise RuntimeError(f"Hooked activation lacks terminal-patch positions: {tuple(post_block_output.shape)}")
        changed = post_block_output.expand(len(chunk), -1, -1).clone()
        changed[:, TERMINAL_PATCH_START:TERMINAL_PATCH_END, :] = replacements.to(device=changed.device, dtype=changed.dtype)
        with torch.no_grad():
            output = reexecute_post_block_tail(model, changed).detach()
        stability, drops = patch_metrics_terminal(clean_representation, output)
        for item, value, drop in zip(chunk, stability, drops):
            results.append({**item, "cosine_stability": value, "percentage_drop": drop})
    return results


def metric_row_terminal(
    record: dict[str, Any],
    condition: str,
    stability: float,
    drop: float,
    fidelity: dict[str, float],
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
        "intervened_positions": "[257:261]",
        "true_register_positions": "[1:5] (excluded from both intervention and metric here)",
        "patch_metric_positions": "[5:257]",
        "sae_scope": "released registers_only checkpoint; trained on terminal patches [257:261), in-distribution here",
        "feature_selection_rule": "five largest positive active (terminal-patch position, latent ID) coordinates per image",
        "cosine_stability": stability,
        "percentage_drop": drop,
        "reconstruction_mse": fidelity["mse"],
        "reconstruction_cosine": fidelity["cosine"],
        "reconstruction_explained_variance": fidelity["explained_variance"],
        "selected_top5_active": json.dumps(selected),
        "seed": SEED,
    }


def evaluate_image(
    record: dict[str, Any],
    cohort: str,
    model: torch.nn.Module,
    module: torch.nn.Module,
    processor: Any,
    sae: AutoEncoderTopK,
    random_draws: int,
    intervention_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    validate_identity: bool,
    model_name: str = MODEL_NAME,
    sae_path: str = str(MISTARGETED_SAE / "ae.pt"),
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
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
    if tuple(full_activation.shape[1:]) != (SEQUENCE_LENGTH, sae.activation_dim):
        raise RuntimeError(f"Unexpected hooked output shape {tuple(full_activation.shape)}")
    terminal = full_activation[0, TERMINAL_PATCH_START:TERMINAL_PATCH_END].to(dtype=torch.float32)

    with torch.no_grad():
        xhat, features = encode_decode(sae, terminal)
    mse, cosine, fve = reconstruction_stats(terminal, xhat)
    fidelity = {"mse": mse, "cosine": cosine, "explained_variance": fve}
    fidelity_row = {
        "image_id": record["source_id"], "stream_position": record["stream_position"], "label": record["label"], "split": record["split"],
        "sae": "registers_only checkpoint (mislabeled)", "sae_path": sae_path,
        "evaluated_positions": "terminal patches [257:261)", "why": "population the released selector actually trained on",
        "reconstruction_mse": mse, "reconstruction_cosine": cosine, "explained_variance": fve,
    }

    selected = coordinates_from_positive_support(features)
    top_contribution = selected_contribution(sae, features, selected)
    masked_features = features.clone()
    for row, latent in selected:
        masked_features[row, latent] = 0
    masked_decode_error = float((sae.decode(masked_features) - (xhat - top_contribution)).abs().max().item())
    if masked_decode_error > TOL:
        raise AssertionError(f"Masked decode mismatch: {masked_decode_error}")
    selected_rows = {row for row, _ in selected}
    outside_selected_error = max((float(top_contribution[row].abs().max().item()) for row in range(TERMINAL_PATCH_END - TERMINAL_PATCH_START) if row not in selected_rows), default=0.0)
    if outside_selected_error > TOL:
        raise AssertionError(f"Selected contribution leaked outside source terminal-patch position: {outside_selected_error}")

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
        {"condition": "reconstruction", "draw_index": None, "draw_count": None, "control_seed": None, "replacement": xhat},
        {"condition": "clean_top5", "draw_index": None, "draw_count": None, "control_seed": None, "replacement": terminal - top_contribution},
        {"condition": "clean_sign_flip", "draw_index": None, "draw_count": None, "control_seed": None, "replacement": terminal + top_contribution},
    ]
    span_audit: list[dict[str, Any]] = []
    permutation_audit: list[dict[str, Any]] = []
    invalid_span = False
    invalid_permutation = False
    for draw_index in range(random_draws):
        span_rng, span_seed = deterministic_rng(SEED, cohort, record["stream_position"], "random_span", draw_index)
        span_candidate, span_ids, span_coefficients = random_decoder_span(sae, features, span_rng)
        span_scaled, target_norm, span_unscaled_norm, span_achieved_norm = match_energy(span_candidate, top_contribution)
        span_zero_rows = [row for row, (target, candidate) in enumerate(zip(target_norm, span_unscaled_norm)) if target > 1e-12 and candidate <= 1e-12]
        if span_zero_rows:
            invalid_span = True
        for target, achieved in zip(target_norm, span_achieved_norm):
            if target > 1e-12 and abs(target - achieved) > 5e-4:
                raise AssertionError("Random span energy match failed")
        interventions.append({"condition": "clean_random_span_draw", "draw_index": draw_index, "draw_count": random_draws, "control_seed": span_seed, "replacement": terminal - span_scaled})
        span_audit.append({"draw_index": draw_index, "seed": span_seed, "decoder_latent_ids_per_row": span_ids, "normal_coefficients_per_row": span_coefficients, "target_norm_per_row": target_norm, "candidate_unscaled_norm_per_row": span_unscaled_norm, "achieved_norm_per_row": span_achieved_norm, "zero_norm_rows": span_zero_rows})

        permutation_rng, permutation_seed = deterministic_rng(SEED, cohort, record["stream_position"], "decoder_permutation", draw_index)
        permutation_candidate, mapping = decoder_permutation_control(sae, features, selected, permutation_rng)
        permutation_scaled, permutation_target_norm, permutation_unscaled_norm, permutation_achieved_norm = match_energy(permutation_candidate, top_contribution)
        permutation_zero_rows = [row for row, (target, candidate) in enumerate(zip(permutation_target_norm, permutation_unscaled_norm)) if target > 1e-12 and candidate <= 1e-12]
        if permutation_zero_rows:
            invalid_permutation = True
        for target, achieved in zip(permutation_target_norm, permutation_achieved_norm):
            if target > 1e-12 and abs(target - achieved) > 5e-4:
                raise AssertionError("Decoder-permutation energy match failed")
        interventions.append({"condition": "clean_decoder_permutation_draw", "draw_index": draw_index, "draw_count": random_draws, "control_seed": permutation_seed, "replacement": terminal - permutation_scaled})
        permutation_audit.append({"draw_index": draw_index, "seed": permutation_seed, "support_and_magnitude_mapping": mapping, "target_norm_per_row": permutation_target_norm, "candidate_unscaled_norm_per_row": permutation_unscaled_norm, "achieved_norm_per_row": permutation_achieved_norm, "zero_norm_rows": permutation_zero_rows})

    intervention_results = batched_interventions_terminal(model, full_activation, clean_output, interventions, intervention_batch_size)
    common_exclusions: list[str] = []
    if invalid_span:
        common_exclusions.append("clean_random_span_mean")
    if invalid_permutation:
        common_exclusions.append("clean_decoder_permutation_mean")
    rows = [metric_row_terminal(record, "clean", 1.0, 0.0, fidelity, selected, common_exclusions, model_name=model_name)]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for result in intervention_results:
        row = metric_row_terminal(
            record, result["condition"], result["cosine_stability"], result["percentage_drop"],
            fidelity, selected, common_exclusions,
            draw_index=result["draw_index"], draw_count=result["draw_count"], control_seed=result["control_seed"],
            model_name=model_name,
        )
        rows.append(row)
        by_condition.setdefault(result["condition"], []).append(row)
    for raw_condition, mean_condition in [
        ("clean_random_span_draw", "clean_random_span_mean"),
        ("clean_decoder_permutation_draw", "clean_decoder_permutation_mean"),
    ]:
        raw_rows = by_condition[raw_condition]
        rows.append(metric_row_terminal(
            record, mean_condition,
            float(np.mean([row["cosine_stability"] for row in raw_rows])),
            float(np.mean([row["percentage_drop"] for row in raw_rows])),
            fidelity, selected, common_exclusions, draw_count=random_draws,
            model_name=model_name,
        ))
    audit = {
        "image_id": record["source_id"],
        "stream_position": record["stream_position"],
        "eligible": True,
        "top5_active": selected,
        "top5_positive_coefficients": [float(features[row, latent].item()) for row, latent in selected],
        "top5_norm_per_row": torch.linalg.vector_norm(top_contribution, dim=-1).detach().cpu().tolist(),
        "identity_hook_max_abs_error": identity_error,
        "manual_post_block_tail_max_abs_error": tail_clean_error,
        "masked_decode_max_abs_error": masked_decode_error,
        "top5_nonselected_row_max_abs": outside_selected_error,
        "random_span_draws": span_audit,
        "decoder_permutation_draws": permutation_audit,
        "pairwise_excluded_conditions": common_exclusions,
    }
    return fidelity_row, rows, audit


def run_cohort(args: argparse.Namespace) -> None:
    if args.cohort == "primary":
        stream_start, default_size = 0, PRIMARY_SIZE
    else:
        stream_start, default_size = HOLDOUT_STREAM_START, HOLDOUT_SIZE
    n_images = args.n_images or default_size
    if n_images != default_size and not args.allow_nonprotocol_size:
        raise ValueError(f"{args.cohort} protocol requires {default_size} images; use --allow-nonprotocol-size only for a smoke test")
    output = args.output_dir or (DEFAULT_OUTPUT / args.cohort)
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
    if args.cohort == "primary" and not args.allow_nonprotocol_size:
        assert_primary_prefix(records, args.p0_output)
    elif args.cohort == "holdout" and not args.allow_nonprotocol_size:
        p0_ids = {item["source_id"] for item in json.loads((args.p0_output / "image_ids.json").read_text(encoding="utf-8"))}
        overlap = p0_ids & {record["source_id"] for record in records}
        if overlap:
            raise AssertionError(f"Fixed holdout overlaps P0 cohort: {sorted(overlap)[:3]}")

    sae_path = args.sae / "ae.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {sae_path}")
    sae = AutoEncoderTopK.from_pretrained(str(sae_path), device=device)
    sae.eval()
    model, _tokenizer, processor = load_model(args.model_name, make_cfg(str(device), dtype, args.model_name), dtype=dtype, device=str(device))
    model.eval()
    module = resolve_attr(model, f"encoder.layer[{BLOCK_INDEX}]")
    save_json(output / "image_ids.json", public_image_records(records))

    config = {
        "command": " ".join(sys.argv),
        "cohort": args.cohort,
        "protocol_cohort_size": default_size,
        "n_selected": len(records),
        "stream_window": [stream_start, stream_start + n_images],
        "cohort_source": f"ILSVRC/imagenet-1k validation streaming shuffle(seed={args.seed}, buffer_size={args.shuffle_buffer})",
        "model_name": args.model_name,
        "model_revision": model_revision(args.model_name),
        "hook": HOOK_LABEL,
        "hook_block_index_zero_based": BLOCK_INDEX,
        "intervened_positions": "[257:261) -- terminal patches, the released registers_only selector's actual training population",
        "true_register_positions_for_contrast": "[1:5) -- excluded from both intervention and metric here",
        "patch_metric_positions": f"[{PATCH_METRIC_START}:{PATCH_METRIC_END}) -- excludes CLS, true registers, and the intervened terminal patches",
        "sae": {"path": str(sae_path), "sha256": sha256_file(sae_path), "config": str(args.sae / "config.json"), "scope": "released registers_only checkpoint; in-distribution at terminal patches"},
        "random_controls": {"random_draws_per_image": args.random_draws, "aggregation": "mean of raw control outcomes within image before image-level paired bootstrap", "random_span": "five random decoder columns with Normal(0,1) coefficients per row, energy-matched per row", "decoder_permutation": "same selected support and positive coefficients, nonselected decoder-column identities sampled without replacement, energy-matched per row"},
        "purpose": "Tests whether the F3 sign reversal (selected top-five removal less disruptive than matched controls) is specific to register-token geometry, or a general property of intervening on a population the SAE was actually trained on. F4 ran legacy_top5/bottom5 at this same population but never ran matched controls here.",
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
        fidelity_row, rows, audit = evaluate_image(
            record, args.cohort, model, module, processor, sae,
            args.random_draws, args.intervention_batch_size, device, dtype, validate_identity=index <= 2,
            model_name=args.model_name, sae_path=str(sae_path),
        )
        fidelity_rows.append(fidelity_row)
        metric_rows.extend(rows)
        audits.append(audit)
        if index % 16 == 0 or index == len(records):
            logging.info("Processed %d/%d images", index, len(records))

    pd.DataFrame(metric_rows).to_csv(output / "per_image_metrics.csv", index=False)
    pd.DataFrame(fidelity_rows).to_csv(output / "reconstruction_fidelity.csv", index=False)
    save_json(output / "selection_audit.json", audits)
    metrics = pd.DataFrame(metric_rows)
    fidelity_frame = pd.DataFrame(fidelity_rows)
    summary = {
        "cohort": args.cohort,
        "n_images": len(records),
        "n_metric_rows": len(metric_rows),
        "validation": {
            "identity_hook_max_abs_error": max((float(a["identity_hook_max_abs_error"]) for a in audits if a["identity_hook_max_abs_error"] is not None), default=None),
            "manual_post_block_tail_max_abs_error": max((float(a["manual_post_block_tail_max_abs_error"]) for a in audits if a["manual_post_block_tail_max_abs_error"] is not None), default=None),
            "masked_decode_max_abs_error": max(float(a["masked_decode_max_abs_error"]) for a in audits),
            "top5_nonselected_row_max_abs": max(float(a["top5_nonselected_row_max_abs"]) for a in audits),
            "random_span_zero_norm_draws": int(sum(len(draw["zero_norm_rows"]) for audit in audits for draw in audit["random_span_draws"])),
            "decoder_permutation_zero_norm_draws": int(sum(len(draw["zero_norm_rows"]) for audit in audits for draw in audit["decoder_permutation_draws"])),
            "patch_metric_excludes_cls_true_registers_and_intervened_terminal_patches": True,
        },
        "reconstruction_explained_variance": value_summary(fidelity_frame["explained_variance"].astype(float).to_numpy(), args.seed + 1),
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
    logging.info("Completed %s cohort in %.1fs", args.cohort, time.time() - start_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["primary", "holdout"], required=True)
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
    parser.add_argument("--sae", type=Path, default=MISTARGETED_SAE)
    parser.add_argument("--model-name", default=MODEL_NAME, help="vision model to hook activations from; must match the scale --sae was trained on")
    args = parser.parse_args()
    if not args.allow_nonprotocol_size and args.random_draws < 8:
        raise ValueError("Protocol requires at least eight independent random draws per image")
    if args.intervention_batch_size <= 0:
        raise ValueError("--intervention-batch-size must be positive")
    run_cohort(args)


if __name__ == "__main__":
    main()
