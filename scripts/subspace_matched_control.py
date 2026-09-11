"""Experiment B -- a subspace-matched (on-manifold) control for register ablation.

The paper's two existing nulls -- an energy-matched random decoder span and a
decoder permutation -- are both *off-manifold* by construction: they perturb the
register by a vector of the right norm drawn essentially at random in R^d, which
lands almost entirely outside the register cloud's principal direction (0.6% of
its energy, against 99.7% for the selected contribution).

That leaves the headline reversal open to a weaker reading: perhaps *any*
on-manifold perturbation of matched norm would be equally undisruptive, and the
SAE selection contributes nothing.  This script tests that directly by adding
controls drawn *inside* the register cloud's own top-r principal subspace:

  subspace_random_r1 / r4   random direction in the top-1 / top-4 PC subspace,
                            energy-matched to the selected contribution
  subspace_aligned_r1       the selected contribution's own projection onto the
                            top PC, rescaled to the selected energy -- the
                            maximally on-manifold removal of the same size

Interpretation is fixed in advance, so the result is not readable after the
fact:

  * If clean top-five removal is indistinguishable from the subspace controls,
    the effect is a property of the activation geometry and the SAE selection is
    incidental.  That is a real finding and it constrains what the paper may
    claim.
  * If removal remains reliably distinguishable, the SAE is selecting something
    the top principal direction alone does not capture.

Both outcomes are reportable; neither is assumed.

GPU notes: one full forward pass per image captures layer 8, then all ~36
intervention conditions for that image are stacked into a single batch and
pushed through blocks 9-11 plus the final layer norm together, so the tail is
evaluated once per image rather than once per condition.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.full_sae_register_audit import (  # noqa: E402
    BLOCK_INDEX,
    MODEL_NAME,
    N_PATCHES,
    PATCH_START,
    REGISTER_END,
    REGISTER_START,
    SEED,
    SEQUENCE_LENGTH,
    coordinates_from_positive_support,
    decoder_permutation_control,
    deterministic_rng,
    encode_decode,
    make_cfg,
    match_energy,
    output_tensor,
    random_decoder_span,
    reexecute_post_block_tail,
    selected_contribution,
)
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from utils.utils import load_model  # noqa: E402

FULL_TOKEN_SAE = REPO_ROOT / "saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_24096850/trainer_0"


def subspace_candidate(
    basis: torch.Tensor, n_rows: int, rng: np.random.Generator, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Random direction inside the span of ``basis`` (d x r), one per register row."""
    r = basis.shape[1]
    coefficients = torch.as_tensor(
        rng.normal(size=(n_rows, r)).astype(np.float32), dtype=dtype, device=device
    )
    return coefficients @ basis.T


def aligned_candidate(basis: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Projection of ``delta`` onto the span of ``basis``, per row."""
    return (delta @ basis) @ basis.T


def paired_bootstrap(
    treatment: np.ndarray, control: np.ndarray, resamples: int, seed: int
) -> dict[str, float]:
    """Paired bootstrap over images of mean(treatment) - mean(control)."""
    difference = treatment - control
    rng = np.random.default_rng(seed)
    n = difference.shape[0]
    draws = rng.integers(0, n, size=(resamples, n))
    means = difference[draws].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "excludes_zero": bool(np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cohort-cache", type=Path, required=True)
    parser.add_argument("--allow-missing", type=int, default=0)
    parser.add_argument("--register-basis", type=Path, required=True, help="npz from Experiment A")
    parser.add_argument("--cohort", default="primary")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--full-sae", type=Path, default=FULL_TOKEN_SAE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.output_dir / "run.log"), logging.StreamHandler()],
        force=True,
    )
    start_time = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(args.device)
    dtype = torch.float32

    from scripts.cohort_local import load_cohort

    records = load_cohort(args.image_ids, args.workdir, args.cohort_cache, args.allow_missing)
    logging.info("cohort=%s n=%d", args.cohort, len(records))

    basis_file = np.load(args.register_basis)
    eigenvectors = basis_file["eigenvectors"]
    if int(basis_file["layer"]) != BLOCK_INDEX:
        raise RuntimeError(f"Basis is for layer {int(basis_file['layer'])}, expected {BLOCK_INDEX}")
    basis_r1 = torch.as_tensor(eigenvectors[:, :1].copy(), dtype=dtype, device=device)
    basis_r4 = torch.as_tensor(eigenvectors[:, :4].copy(), dtype=dtype, device=device)
    logging.info("loaded register basis %s from layer %d", eigenvectors.shape, BLOCK_INDEX)

    full_sae = AutoEncoderTopK.from_pretrained(str(args.full_sae / "ae.pt"), device=device)
    full_sae.eval()
    model, _tok, processor = load_model(
        args.model_name, make_cfg(str(device), dtype, args.model_name), dtype=dtype, device=str(device)
    )
    model.eval()
    module = model.encoder.layer[BLOCK_INDEX]

    condition_names = (
        ["reconstruction", "clean_top5", "clean_sign_flip", "subspace_aligned_r1"]
        + [f"random_span_{i}" for i in range(args.random_draws)]
        + [f"decoder_permutation_{i}" for i in range(args.random_draws)]
        + [f"subspace_random_r1_{i}" for i in range(args.random_draws)]
        + [f"subspace_random_r4_{i}" for i in range(args.random_draws)]
    )

    per_image: list[dict[str, Any]] = []
    alignment_log: list[dict[str, float]] = []

    for index, record in enumerate(records):
        pixels = processor(images=record["image"], return_tensors="pt")["pixel_values"].to(
            device=device, dtype=dtype
        )
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

        registers = act[0, REGISTER_START:REGISTER_END]
        with torch.no_grad():
            xhat, features = encode_decode(full_sae, registers)
            selected = coordinates_from_positive_support(features)
            delta_selected = selected_contribution(full_sae, features, selected)

            deltas: list[torch.Tensor] = []
            # reconstruction baseline: replace registers with their reconstruction
            deltas.append(registers - xhat)
            deltas.append(delta_selected)          # clean_top5 (subtracted)
            deltas.append(-delta_selected)         # clean_sign_flip (added back)

            aligned = aligned_candidate(basis_r1, delta_selected)
            aligned_scaled, _t, _u, _a = match_energy(aligned, delta_selected)
            deltas.append(aligned_scaled)

            for draw in range(args.random_draws):
                rng, _ = deterministic_rng(SEED, args.cohort, record["stream_position"], "random_span", draw)
                candidate, _ids, _coef = random_decoder_span(full_sae, features, rng)
                scaled, _t, _u, _a = match_energy(candidate, delta_selected)
                deltas.append(scaled)
            for draw in range(args.random_draws):
                rng, _ = deterministic_rng(SEED, args.cohort, record["stream_position"], "decoder_permutation", draw)
                candidate, _map = decoder_permutation_control(full_sae, features, selected, rng)
                scaled, _t, _u, _a = match_energy(candidate, delta_selected)
                deltas.append(scaled)
            for label, basis in (("subspace_r1", basis_r1), ("subspace_r4", basis_r4)):
                for draw in range(args.random_draws):
                    rng, _ = deterministic_rng(SEED, args.cohort, record["stream_position"], label, draw)
                    candidate = subspace_candidate(basis, registers.shape[0], rng, dtype, device)
                    scaled, _t, _u, _a = match_energy(candidate, delta_selected)
                    deltas.append(scaled)

            # Record how each control aligns with the selected contribution.
            flat_selected = delta_selected.reshape(-1)
            alignment_log.append({
                "stream_position": record["stream_position"],
                "aligned_r1_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        aligned_scaled.reshape(1, -1), flat_selected.reshape(1, -1)
                    ).item()
                ),
                "selected_energy_in_r1": float(
                    (aligned.reshape(-1) ** 2).sum() / (flat_selected**2).sum()
                ),
            })

            # One batched pass through blocks 9-11 for every condition at once.
            stacked = act.expand(len(deltas), -1, -1).clone()
            for position, delta in enumerate(deltas):
                stacked[position, REGISTER_START:REGISTER_END] -= delta
            outputs = reexecute_post_block_tail(model, torch.cat([act, stacked], dim=0))

        clean_out = outputs[:1, PATCH_START : PATCH_START + N_PATCHES]
        cond_out = outputs[1:, PATCH_START : PATCH_START + N_PATCHES]
        cosine = torch.nn.functional.cosine_similarity(
            clean_out.expand_as(cond_out), cond_out, dim=-1
        ).mean(dim=-1)
        drops = [100.0 * (1.0 - float(v)) for v in cosine.detach().cpu().tolist()]

        row = {"stream_position": record["stream_position"], "source_id": record["source_id"]}
        row.update(dict(zip(condition_names, drops)))
        per_image.append(row)

        if (index + 1) % 64 == 0:
            logging.info("processed %d/%d images (%.1fs)", index + 1, len(records), time.time() - start_time)

    def column(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in per_image], dtype=np.float64)

    def control_mean(prefix: str) -> np.ndarray:
        cols = [column(f"{prefix}_{i}") for i in range(args.random_draws)]
        return np.mean(np.stack(cols, axis=0), axis=0)

    top5 = column("clean_top5")
    controls = {
        "random_span": control_mean("random_span"),
        "decoder_permutation": control_mean("decoder_permutation"),
        "subspace_random_r1": control_mean("subspace_random_r1"),
        "subspace_random_r4": control_mean("subspace_random_r4"),
        "subspace_aligned_r1": column("subspace_aligned_r1"),
    }

    contrasts = {
        name: paired_bootstrap(top5, values, args.bootstrap, SEED + i)
        for i, (name, values) in enumerate(controls.items())
    }

    summary = {
        "cohort": args.cohort,
        "model_name": args.model_name,
        "n_images": len(per_image),
        "random_draws": args.random_draws,
        "bootstrap_resamples": args.bootstrap,
        "register_basis": str(args.register_basis),
        "condition_means": {
            "reconstruction": float(column("reconstruction").mean()),
            "clean_top5": float(top5.mean()),
            "clean_sign_flip": float(column("clean_sign_flip").mean()),
            **{name: float(values.mean()) for name, values in controls.items()},
        },
        "contrasts_top5_minus_control": contrasts,
        "mean_selected_energy_in_top_pc": float(
            np.mean([a["selected_energy_in_r1"] for a in alignment_log])
        ),
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import csv

    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    logging.info("=== condition means (percentage-point drop) ===")
    for name, value in summary["condition_means"].items():
        logging.info("  %-24s %8.4f", name, value)
    logging.info("=== contrasts: clean_top5 minus control ===")
    for name, stats in contrasts.items():
        logging.info(
            "  %-24s %+8.4f  CI95 [%+.4f, %+.4f]  excludes_zero=%s",
            name,
            stats["mean_difference"],
            stats["ci95_low"],
            stats["ci95_high"],
            stats["excludes_zero"],
        )
    logging.info("done in %.1fs", time.time() - start_time)


if __name__ == "__main__":
    main()
