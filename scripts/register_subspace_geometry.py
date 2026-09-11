"""Direct measurement of the register-subspace geometry invoked in Section 5.

Section 5 explains the matched-control sign reversal geometrically: removing the
selected top-five contribution is said to return a register toward the center of
its own low-dimensional subspace, whereas an energy-matched random decoder span
of the same norm displaces it off that subspace. That account has so far been
argued from the sign of the effect, not measured. This script measures it.

For the same cohort, hook, and SAE used by scripts/full_sae_register_audit.py:

  1. Effective dimensionality (participation ratio) of the register activation
     cloud at [1:5) versus the patch cloud at [5:261).
  2. For each image, the fraction of the top-five decoder contribution's energy
     that lies inside the register subspace (top-d principal directions of the
     centered register cloud), versus the same fraction for the energy-matched
     random decoder-span control. The geometric account predicts the selected
     contribution is substantially more in-subspace than the control.
  3. Whether removal actually moves the register toward the cloud center:
     ||r - dtop - mu|| / ||r - mu||, against the same ratio for the control and
     for the sign-flipped (r + dtop) perturbation. Values below 1 mean the
     perturbation moved the register closer to the center.

Subspace dimension d is reported over a sweep rather than fixed, so the
conclusion does not rest on one arbitrary cutoff.

Usage mirrors the audit script:
  python scripts/register_subspace_geometry.py --cohort primary --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from full_sae_register_audit import (  # noqa: E402
    BLOCK_INDEX,
    FULL_TOKEN_SAE,
    HOLDOUT_SIZE,
    HOLDOUT_STREAM_START,
    MODEL_NAME,
    PATCH_START,
    PRIMARY_SIZE,
    REGISTER_END,
    REGISTER_START,
    SEED,
    SEQUENCE_LENGTH,
    bootstrap_ci,
    code_revision,
    coordinates_from_positive_support,
    deterministic_rng,
    encode_decode,
    hardware_metadata,
    make_cfg,
    match_energy,
    model_revision,
    public_image_records,
    random_decoder_span,
    save_json,
    select_dataset_window,
    selected_contribution,
    sha256_file,
)
from intervention_controls import output_tensor  # noqa: E402
from utils.utils import load_model, resolve_attr  # noqa: E402
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402

SUBSPACE_DIMS = [1, 2, 3, 4, 8, 16, 32, 64]


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """(sum l)^2 / sum l^2 -- the standard effective-dimensionality summary."""
    positive = eigenvalues[eigenvalues > 0]
    if positive.size == 0:
        return 0.0
    return float(positive.sum() ** 2 / (positive**2).sum())


def energy_fraction_in_subspace(delta: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of ||delta||^2 captured by projection onto the columns of basis."""
    total = float((delta**2).sum())
    if total <= 1e-20:
        return float("nan")
    projected = basis.T @ delta
    return float((projected**2).sum() / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["primary", "holdout"], default="primary")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--full-sae", type=Path, default=FULL_TOKEN_SAE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stream_start, n_images = (0, PRIMARY_SIZE) if args.cohort == "primary" else (HOLDOUT_STREAM_START, HOLDOUT_SIZE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.output_dir / "run.log"), logging.StreamHandler()],
        force=True,
    )
    start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    records = select_dataset_window(stream_start, n_images, args.seed, args.shuffle_buffer)
    logging.info("cohort=%s n=%d", args.cohort, len(records))

    sae_path = args.full_sae / "ae.pt"
    full_sae = AutoEncoderTopK.from_pretrained(str(sae_path), device=device)
    full_sae.eval()
    model, _tok, processor = load_model(args.model_name, make_cfg(str(device), dtype, args.model_name), dtype=dtype, device=str(device))
    model.eval()
    module = resolve_attr(model, f"encoder.layer[{BLOCK_INDEX}]")

    registers_all: list[np.ndarray] = []
    patches_all: list[np.ndarray] = []
    per_image: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        pixels = processor(images=record["image"], return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
        captured: dict[str, torch.Tensor] = {}

        def capture(_m, _i, output):
            captured["activation"] = output_tensor(output).detach()
            return output

        handle = module.register_forward_hook(capture)
        with torch.no_grad():
            model(pixel_values=pixels)
        handle.remove()
        act = captured["activation"]
        if tuple(act.shape[1:]) != (SEQUENCE_LENGTH, full_sae.activation_dim):
            raise RuntimeError(f"Unexpected hooked shape {tuple(act.shape)}")

        registers = act[0, REGISTER_START:REGISTER_END].to(dtype=torch.float32)
        patches = act[0, PATCH_START:SEQUENCE_LENGTH].to(dtype=torch.float32)
        registers_all.append(registers.cpu().numpy())
        patches_all.append(patches.cpu().numpy())

        with torch.no_grad():
            _xhat, features = encode_decode(full_sae, registers)
            selected = coordinates_from_positive_support(features)
            top_contribution = selected_contribution(full_sae, features, selected)

            # Energy-matched random-span controls, same construction and seeds as the audit.
            control_deltas: list[np.ndarray] = []
            for draw_index in range(args.random_draws):
                span_rng, _seed = deterministic_rng(SEED, args.cohort, record["stream_position"], "random_span", draw_index)
                span_candidate, _ids, _coef = random_decoder_span(full_sae, features, span_rng)
                span_scaled, _t, _u, _a = match_energy(span_candidate, top_contribution)
                control_deltas.append(span_scaled.detach().cpu().numpy())

        per_image.append({
            "source_id": record["source_id"],
            "stream_position": record["stream_position"],
            "registers": registers.detach().cpu().numpy(),
            "delta_top": top_contribution.detach().cpu().numpy(),
            "delta_controls": control_deltas,
        })
        if (index + 1) % 64 == 0:
            logging.info("processed %d/%d", index + 1, len(records))

    register_cloud = np.concatenate(registers_all, axis=0).astype(np.float64)
    patch_cloud = np.concatenate(patches_all, axis=0).astype(np.float64)
    logging.info("register cloud %s, patch cloud %s", register_cloud.shape, patch_cloud.shape)

    mu_register = register_cloud.mean(axis=0)
    centered = register_cloud - mu_register
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    mu_patch = patch_cloud.mean(axis=0)
    centered_patch = patch_cloud - mu_patch
    cov_patch = (centered_patch.T @ centered_patch) / max(centered_patch.shape[0] - 1, 1)
    eigenvalues_patch = np.linalg.eigvalsh(cov_patch)[::-1]

    total_variance = float(eigenvalues[eigenvalues > 0].sum())
    dimensionality = {
        "activation_dim": int(register_cloud.shape[1]),
        "n_register_vectors": int(register_cloud.shape[0]),
        "n_patch_vectors": int(patch_cloud.shape[0]),
        "register_participation_ratio": participation_ratio(eigenvalues),
        "patch_participation_ratio": participation_ratio(eigenvalues_patch),
        "register_cumulative_variance_at_dim": {
            str(d): float(eigenvalues[:d][eigenvalues[:d] > 0].sum() / total_variance) for d in SUBSPACE_DIMS
        },
    }
    logging.info("register PR=%.2f  patch PR=%.2f", dimensionality["register_participation_ratio"], dimensionality["patch_participation_ratio"])

    results: dict[str, Any] = {}
    for d in SUBSPACE_DIMS:
        basis = eigenvectors[:, :d]
        top_fracs, control_fracs, paired = [], [], []
        for item in per_image:
            for row in range(item["delta_top"].shape[0]):
                dtop = item["delta_top"][row].astype(np.float64)
                if float((dtop**2).sum()) <= 1e-20:
                    continue
                ftop = energy_fraction_in_subspace(dtop, basis)
                fctrl = float(np.mean([
                    energy_fraction_in_subspace(ctrl[row].astype(np.float64), basis)
                    for ctrl in item["delta_controls"]
                ]))
                top_fracs.append(ftop)
                control_fracs.append(fctrl)
                paired.append(ftop - fctrl)
        top_arr, ctrl_arr, paired_arr = map(lambda v: np.asarray(v, dtype=np.float64), (top_fracs, control_fracs, paired))
        results[str(d)] = {
            "n": int(top_arr.size),
            "selected_in_subspace_mean": float(top_arr.mean()),
            "selected_in_subspace_ci95": bootstrap_ci(top_arr, args.seed + d),
            "control_in_subspace_mean": float(ctrl_arr.mean()),
            "control_in_subspace_ci95": bootstrap_ci(ctrl_arr, args.seed + d + 1),
            "paired_difference_mean": float(paired_arr.mean()),
            "paired_difference_ci95": bootstrap_ci(paired_arr, args.seed + d + 2),
        }
        logging.info("d=%-3d selected=%.4f control=%.4f diff=%.4f", d, top_arr.mean(), ctrl_arr.mean(), paired_arr.mean())

    # Does removal move the register toward the center of its own cloud?
    ratio_removal, ratio_control, ratio_signflip = [], [], []
    for item in per_image:
        for row in range(item["registers"].shape[0]):
            r = item["registers"][row].astype(np.float64)
            dtop = item["delta_top"][row].astype(np.float64)
            if float((dtop**2).sum()) <= 1e-20:
                continue
            base = float(np.linalg.norm(r - mu_register))
            if base <= 1e-12:
                continue
            ratio_removal.append(float(np.linalg.norm(r - dtop - mu_register)) / base)
            ratio_signflip.append(float(np.linalg.norm(r + dtop - mu_register)) / base)
            ratio_control.append(float(np.mean([
                np.linalg.norm(r - ctrl[row].astype(np.float64) - mu_register) / base
                for ctrl in item["delta_controls"]
            ])))
    centering = {}
    for name, values in [("removal", ratio_removal), ("random_span_control", ratio_control), ("sign_flip", ratio_signflip)]:
        arr = np.asarray(values, dtype=np.float64)
        centering[name] = {
            "n": int(arr.size),
            "distance_to_center_ratio_mean": float(arr.mean()),
            "distance_to_center_ratio_ci95": bootstrap_ci(arr, args.seed + 77),
            "fraction_moved_closer_to_center": float((arr < 1.0).mean()),
        }
        logging.info("%-20s distance ratio %.4f  moved closer %.1f%%", name, arr.mean(), 100 * (arr < 1.0).mean())

    summary = {
        "cohort": args.cohort,
        "model_name": args.model_name,
        "model_revision": model_revision(args.model_name),
        "code_revision": code_revision(),
        "hook": f"enc_res_out_layer_{BLOCK_INDEX}",
        "sae_path": str(sae_path),
        "sae_sha256": sha256_file(sae_path),
        "seed": args.seed,
        "random_draws": args.random_draws,
        "hardware": hardware_metadata(),
        "wall_time_seconds": round(time.time() - start, 1),
        "dimensionality": dimensionality,
        "subspace_energy_by_dim": results,
        "centering": centering,
        "notes": (
            "selected_in_subspace / control_in_subspace are fractions of squared perturbation norm lying in the "
            "top-d principal subspace of the centered register cloud. distance_to_center_ratio is "
            "||r + delta - mu|| / ||r - mu||; below 1 means the perturbation moved the register toward the cloud center. "
            "Controls are energy-matched random decoder spans drawn with the same seeds as the main audit."
        ),
    }
    save_json(args.output_dir / "summary.json", summary)
    save_json(args.output_dir / "image_ids.json", public_image_records(records))
    np.save(args.output_dir / "register_eigenvalues.npy", eigenvalues)
    logging.info("wrote %s (%.1fs)", args.output_dir / "summary.json", summary["wall_time_seconds"])


if __name__ == "__main__":
    main()
