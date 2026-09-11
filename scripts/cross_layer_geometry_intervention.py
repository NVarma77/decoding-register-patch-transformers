"""Experiment C -- does the ablation reversal track register dimensionality?

Experiment A found that the register cloud is not one-dimensional at every
depth.  It arrives with participation ratio ~3 (layers 0-3, top PC carrying only
40-53% of variance), collapses to near rank-one at layer 4 (PR 1.001, top PC
99.94%), and re-expands by layer 11 (PR 1.915).

If the reversal reported at layer 8 is caused by that near-rank-one geometry --
removal moves the register *along* its own occupied direction, toward the
population mean, while an equal-norm random perturbation moves it off into
unoccupied space -- then the reversal is a prediction, not a description, and it
should weaken where the geometry is not degenerate.  This script tests that.

Why no SAE
----------
Full-token SAEs exist only at layer 8 in this repo, so a top-k SAE ablation at
layer 2 would be out of distribution and would measure the SAE's failure to
generalise rather than the geometry.  The mechanism itself does not need an SAE:
it is a claim about on-manifold versus off-manifold perturbation.  So the test is
run SAE-free at every layer, with the perturbation defined geometrically:

  on_manifold      remove the register's component along the top-r principal
                   directions of its own cloud (moves it toward the population
                   mean, mirroring what top-5 removal does at layer 8)
  off_manifold     an energy-matched random direction in R^d
  on_manifold_rand an energy-matched random direction *inside* the top-r subspace

This separates the geometric law from the SAE question.  Experiment B asks
whether the SAE adds anything at layer 8; this asks whether the law holds at all.

Prediction, fixed before running: the on-manifold advantage (off_manifold minus
on_manifold, positive = removal less disruptive) should be large at layers 4-9
and markedly smaller at layers 0-3.  A flat profile across depth would falsify
the geometric account and would mean the layer-8 result needs a different
explanation.
"""

from __future__ import annotations

import argparse
import csv
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
    MODEL_NAME,
    N_PATCHES,
    PATCH_START,
    REGISTER_END,
    REGISTER_START,
    SEED,
    SEQUENCE_LENGTH,
    deterministic_rng,
    make_cfg,
    match_energy,
    output_tensor,
)
from scripts.subspace_matched_control import paired_bootstrap  # noqa: E402
from utils.utils import load_model  # noqa: E402


def tail_from_layer(model: torch.nn.Module, hidden: torch.Tensor, layer: int) -> torch.Tensor:
    """Re-execute every block after ``layer`` plus the final layer norm."""
    for index in range(layer + 1, len(model.encoder.layer)):
        hidden = model.encoder.layer[index](hidden, None, False)[0]
    return model.layernorm(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cohort-cache", type=Path, required=True)
    parser.add_argument("--allow-missing", type=int, default=0)
    parser.add_argument("--bases", type=Path, required=True, help="register_bases_all_layers.npz")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8, 10, 11])
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--cohort", default="primary")
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
    logging.info("cohort=%s n=%d layers=%s rank=%d", args.cohort, len(records), args.layers, args.rank)

    bases = np.load(args.bases)
    model, _tok, processor = load_model(
        args.model_name, make_cfg(str(device), dtype, args.model_name), dtype=dtype, device=str(device)
    )
    model.eval()

    basis: dict[int, torch.Tensor] = {}
    mean: dict[int, torch.Tensor] = {}
    for layer in args.layers:
        basis[layer] = torch.as_tensor(
            bases[f"eigenvectors_{layer}"][:, : args.rank].copy(), dtype=dtype, device=device
        )
        mean[layer] = torch.as_tensor(bases[f"mean_{layer}"].copy(), dtype=dtype, device=device)

    conditions = (
        ["on_manifold"]
        + [f"off_manifold_{i}" for i in range(args.random_draws)]
        + [f"on_manifold_rand_{i}" for i in range(args.random_draws)]
    )

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        pixels = processor(images=record["image"], return_tensors="pt")["pixel_values"].to(
            device=device, dtype=dtype
        )
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in args.layers:
            handles.append(
                model.encoder.layer[layer].register_forward_hook(
                    lambda _m, _i, o, l=layer: captured.__setitem__(l, output_tensor(o).detach())
                )
            )
        with torch.no_grad():
            model(pixel_values=pixels)
        for handle in handles:
            handle.remove()

        row: dict[str, Any] = {
            "stream_position": record["stream_position"],
            "source_id": record["source_id"],
        }
        with torch.no_grad():
            for layer in args.layers:
                act = captured[layer]
                registers = act[0, REGISTER_START:REGISTER_END]
                V, mu = basis[layer], mean[layer]

                # On-manifold removal: strip the component along the cloud's own
                # top-r directions, which moves the register toward the mean.
                centered = registers - mu
                delta_on = (centered @ V) @ V.T

                deltas = [delta_on]
                for draw in range(args.random_draws):
                    rng, _ = deterministic_rng(
                        SEED, args.cohort, record["stream_position"], f"L{layer}_off", draw
                    )
                    cand = torch.as_tensor(
                        rng.normal(size=registers.shape).astype(np.float32), dtype=dtype, device=device
                    )
                    scaled, _t, _u, _a = match_energy(cand, delta_on)
                    deltas.append(scaled)
                for draw in range(args.random_draws):
                    rng, _ = deterministic_rng(
                        SEED, args.cohort, record["stream_position"], f"L{layer}_onrand", draw
                    )
                    coef = torch.as_tensor(
                        rng.normal(size=(registers.shape[0], args.rank)).astype(np.float32),
                        dtype=dtype, device=device,
                    )
                    scaled, _t, _u, _a = match_energy(coef @ V.T, delta_on)
                    deltas.append(scaled)

                stacked = act.expand(len(deltas), -1, -1).clone()
                for position, delta in enumerate(deltas):
                    stacked[position, REGISTER_START:REGISTER_END] -= delta
                outputs = tail_from_layer(model, torch.cat([act, stacked], dim=0), layer)

                clean = outputs[:1, PATCH_START : PATCH_START + N_PATCHES]
                other = outputs[1:, PATCH_START : PATCH_START + N_PATCHES]
                cosine = torch.nn.functional.cosine_similarity(
                    clean.expand_as(other), other, dim=-1
                ).mean(dim=-1)
                drops = [100.0 * (1.0 - float(v)) for v in cosine.detach().cpu().tolist()]
                for name, value in zip(conditions, drops):
                    row[f"L{layer}_{name}"] = value
        rows.append(row)
        if (index + 1) % 64 == 0:
            logging.info("processed %d/%d (%.0fs)", index + 1, len(records), time.time() - start_time)

    def col(name: str) -> np.ndarray:
        return np.asarray([r[name] for r in rows], dtype=np.float64)

    per_layer: list[dict[str, Any]] = []
    for layer in args.layers:
        on = col(f"L{layer}_on_manifold")
        off = np.mean(np.stack([col(f"L{layer}_off_manifold_{i}") for i in range(args.random_draws)]), axis=0)
        onr = np.mean(np.stack([col(f"L{layer}_on_manifold_rand_{i}") for i in range(args.random_draws)]), axis=0)
        # Positive advantage = on-manifold removal is LESS disruptive than the
        # energy-matched off-manifold control (the reversal, restated).
        advantage = paired_bootstrap(off, on, args.bootstrap, SEED + layer)
        entry = {
            "layer": layer,
            "on_manifold_mean": float(on.mean()),
            "off_manifold_mean": float(off.mean()),
            "on_manifold_random_mean": float(onr.mean()),
            "on_manifold_advantage": advantage,
        }
        per_layer.append(entry)
        logging.info(
            "layer %2d  on=%8.4f  off=%8.4f  advantage=%+8.4f CI[%+.4f,%+.4f] excl0=%s",
            layer, entry["on_manifold_mean"], entry["off_manifold_mean"],
            advantage["mean_difference"], advantage["ci95_low"], advantage["ci95_high"],
            advantage["excludes_zero"],
        )

    summary = {
        "model_name": args.model_name,
        "cohort": args.cohort,
        "n_images": len(rows),
        "rank": args.rank,
        "random_draws": args.random_draws,
        "bootstrap_resamples": args.bootstrap,
        "per_layer": per_layer,
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("done in %.1fs", time.time() - start_time)


if __name__ == "__main__":
    main()
