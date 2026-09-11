#!/usr/bin/env python3
"""Aggregate the P1 corrected-control profile without treating layers as replications.

Layer 8 is the first 256 images from the completed P0 cohort. Layers 4 and 11
are independently run P1 subdirectories. This script checks that all three
runs use precisely the same ordered source IDs before writing a compact,
reproducible appendix artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_P0 = REPO / "outputs/experiments/intervention_controls"
DEFAULT_OUTPUT = DEFAULT_P0 / "layer_robustness"
CONDITIONS = ("clean", "reconstruction", "clean_top5", "clean_random_span")
LAYER_ORDER = (4, 8, 11)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable-no-git-checkout"


def hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count(),
    }


def bootstrap_ci(values: np.ndarray, seed: int, resamples: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def values_summary(values: np.ndarray, seed: int, resamples: int) -> dict[str, Any]:
    if len(values) == 0:
        return {"n_images": 0, "mean": None, "median": None, "bootstrap_ci95": [None, None], "values": []}
    return {
        "n_images": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "bootstrap_ci95": bootstrap_ci(values, seed, resamples),
        "values": values.tolist(),
    }


def exclusions(row: dict[str, Any]) -> set[str]:
    value = row.get("pairwise_excluded_conditions", [])
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, str):
        return set(json.loads(value))
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return set(value)
    raise TypeError(f"Unsupported pairwise exclusion value: {value!r}")


def ordered_ids(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {path}, got {type(records).__name__}")
    return records


def assert_same_cohort(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str) -> None:
    if len(candidate) != len(reference):
        raise AssertionError(f"{label}: expected {len(reference)} image IDs, found {len(candidate)}")
    keys = ("stream_position", "source_id", "label", "split")
    mismatches = []
    for index, (a, b) in enumerate(zip(reference, candidate)):
        if any(a.get(key) != b.get(key) for key in keys):
            mismatches.append({"index": index, "expected": {key: a.get(key) for key in keys}, "actual": {key: b.get(key) for key in keys}})
            if len(mismatches) == 3:
                break
    if mismatches:
        raise AssertionError(f"{label}: cohort mismatch: {mismatches}")


def read_metrics(path: Path) -> pd.DataFrame:
    parquet = path / "per_image_metrics.parquet"
    csv_path = path / "per_image_metrics.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No metrics file under {path}")


def load_layer_rows(run_dir: Path, layer: int, source_ids: list[dict[str, Any]], p0_source: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if int(config["layer"]) != layer:
        raise AssertionError(f"{run_dir}: expected layer {layer}, config says {config['layer']}")
    frame = read_metrics(run_dir)
    if p0_source:
        frame = frame[frame["stream_position"].astype(int) < len(source_ids)]
    frame = frame[frame["condition"].isin(CONDITIONS)].copy()
    if len(frame) != len(source_ids) * len(CONDITIONS):
        raise AssertionError(f"{run_dir}: expected {len(source_ids) * len(CONDITIONS)} P1 rows, found {len(frame)}")
    expected = {(str(rec["source_id"]), condition) for rec in source_ids for condition in CONDITIONS}
    observed = {(str(row.image_id), str(row.condition)) for row in frame.itertuples(index=False)}
    if observed != expected:
        missing = sorted(expected - observed)[:3]
        extra = sorted(observed - expected)[:3]
        raise AssertionError(f"{run_dir}: rows do not match cohort; missing={missing}, extra={extra}")
    frame["layer"] = layer
    frame["p1_source"] = "P0 first 256-image prefix" if p0_source else f"P1 layer-{layer} run"
    frame = frame.sort_values(["stream_position", "condition"], kind="stable")
    return frame.to_dict(orient="records"), config


def paired_values(rows: list[dict[str, Any]], left: str, right: str) -> np.ndarray:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["image_id"]), {})[str(row["condition"])] = row
    values: list[float] = []
    for by_condition in by_id.values():
        if left not in by_condition or right not in by_condition:
            continue
        left_row, right_row = by_condition[left], by_condition[right]
        if not bool(left_row.get("eligible", True)) or not bool(right_row.get("eligible", True)):
            continue
        if left in exclusions(right_row) or right in exclusions(left_row):
            continue
        values.append(float(left_row["percentage_drop"]) - float(right_row["percentage_drop"]))
    return np.asarray(values, dtype=np.float64)


def effect_values(rows: list[dict[str, Any]], condition: str) -> np.ndarray:
    return np.asarray([
        float(row["percentage_drop"])
        for row in rows
        if row["condition"] == condition and bool(row.get("eligible", True))
    ], dtype=np.float64)


def fidelity_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    one_per_image: dict[str, float] = {}
    for row in rows:
        if row["condition"] == "clean" and bool(row.get("eligible", True)):
            one_per_image[str(row["image_id"])] = float(row[field])
    return np.asarray(list(one_per_image.values()), dtype=np.float64)


def layer_summary(rows: list[dict[str, Any]], layer: int, seed: int, resamples: int, config: dict[str, Any]) -> dict[str, Any]:
    reconstruction = effect_values(rows, "reconstruction")
    top = effect_values(rows, "clean_top5")
    span = effect_values(rows, "clean_random_span")
    recon_minus_clean = paired_values(rows, "reconstruction", "clean")
    top_minus_span = paired_values(rows, "clean_top5", "clean_random_span")
    return {
        "layer": layer,
        "sae_path": config["sae_path"],
        "sae_sha256": config["sae_sha256"],
        "n_images": int(len(reconstruction)),
        "reconstruction_fidelity": {
            "mse": values_summary(fidelity_values(rows, "reconstruction_mse"), seed + layer * 100 + 1, resamples),
            "cosine": values_summary(fidelity_values(rows, "reconstruction_cosine"), seed + layer * 100 + 2, resamples),
            "explained_variance": values_summary(fidelity_values(rows, "reconstruction_explained_variance"), seed + layer * 100 + 3, resamples),
        },
        "condition_effects_percentage_drop": {
            "reconstruction": values_summary(reconstruction, seed + layer * 100 + 4, resamples),
            "clean_top5": values_summary(top, seed + layer * 100 + 5, resamples),
            "clean_random_span": values_summary(span, seed + layer * 100 + 6, resamples),
        },
        "paired_contrasts_percentage_drop": {
            "reconstruction - clean": values_summary(recon_minus_clean, seed + layer * 100 + 7, resamples),
            "clean_top5 - clean_random_span": values_summary(top_minus_span, seed + layer * 100 + 8, resamples),
        },
    }


def render_figure(summaries: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    layers = [summary["layer"] for summary in summaries]
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.55), constrained_layout=True)
    colors = {"reconstruction": "#777777", "clean_top5": "#1b9e77", "clean_random_span": "#66a61e"}
    labels = {"reconstruction": "reconstruction", "clean_top5": "clean top5", "clean_random_span": "random span"}
    offsets = {"reconstruction": -0.22, "clean_top5": 0.0, "clean_random_span": 0.22}
    for condition in ("reconstruction", "clean_top5", "clean_random_span"):
        values = [summary["condition_effects_percentage_drop"][condition] for summary in summaries]
        means = np.asarray([item["mean"] for item in values])
        ci = np.asarray([item["bootstrap_ci95"] for item in values])
        axes[0].errorbar(x + offsets[condition], means, yerr=np.asarray([means - ci[:, 0], ci[:, 1] - means]), fmt="o", capsize=3, color=colors[condition], label=labels[condition])
    axes[0].set_xticks(x, [f"layer {layer}" for layer in layers])
    axes[0].set_ylabel("Final patch cosine drop (%)")
    axes[0].set_title("Corrected-control effects")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    paired = [summary["paired_contrasts_percentage_drop"]["clean_top5 - clean_random_span"] for summary in summaries]
    means = np.asarray([item["mean"] for item in paired])
    ci = np.asarray([item["bootstrap_ci95"] for item in paired])
    y = np.arange(len(layers))
    axes[1].axvline(0, color="#999999", linewidth=0.8)
    axes[1].errorbar(means, y, xerr=np.asarray([means - ci[:, 0], ci[:, 1] - means]), fmt="o", capsize=3, color="#1b9e77")
    axes[1].set_yticks(y, [f"layer {layer}" for layer in layers])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Top5 minus random span (%)")
    axes[1].set_title("Image-level paired contrast")
    fig.savefig(output / "figure_layer_robustness.png", dpi=220)
    fig.savefig(output / "figure_layer_robustness.pdf")
    plt.close(fig)


def outcome_memo(summaries: list[dict[str, Any]]) -> str:
    by_layer = {item["layer"]: item for item in summaries}
    lines = [
        "# P1 layer-robustness outcome",
        "",
        "Do not promote a main-text layer-robustness claim. The corrected top-minus-random-span ordering is not consistent across the non-terminal layers, and the layer-11 post-block intervention is structurally unable to change downstream patch tokens.",
        "",
        "| Layer | Reconstruction drop (95% CI) | Clean top5 drop | Random-span drop | Top5 minus span (95% CI) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for layer in LAYER_ORDER:
        item = by_layer[layer]
        rec = item["condition_effects_percentage_drop"]["reconstruction"]
        top = item["condition_effects_percentage_drop"]["clean_top5"]
        span = item["condition_effects_percentage_drop"]["clean_random_span"]
        diff = item["paired_contrasts_percentage_drop"]["clean_top5 - clean_random_span"]
        lines.append(f"| {layer} | {rec['mean']:.2f} [{rec['bootstrap_ci95'][0]:.2f}, {rec['bootstrap_ci95'][1]:.2f}] | {top['mean']:.2f} | {span['mean']:.2f} | {diff['mean']:.2f} [{diff['bootstrap_ci95'][0]:.2f}, {diff['bootstrap_ci95'][1]:.2f}] |")
    lines.extend([
        "",
        "The direct two-image terminal-layer locality check changed final-block register outputs by more than 40 in max absolute value but changed final patch outputs by 0.0. See `layer_11/terminal_layer_locality_check.json`. This is a hook-placement limitation of this P1 profile, not evidence that the SAE reconstruction is exact at layer 11.",
        "",
        "This is a shared-cohort robustness profile, not three independent replications. Full commands, hardware, source paths, raw image-level records, and 10,000-resample bootstrap summaries are in `config.json`, `per_image_metrics.parquet`, and `summary.json`.",
    ])
    return "\n".join(lines) + "\n"


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "per_image_metrics.parquet", index=False)
    frame.to_csv(output / "per_image_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    parser.add_argument("--layer4-dir", type=Path, default=DEFAULT_OUTPUT / "layer_4")
    parser.add_argument("--layer11-dir", type=Path, default=DEFAULT_OUTPUT / "layer_11")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.n_images <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("--n-images and --bootstrap-resamples must be positive")

    p0_ids_all = ordered_ids(args.p0_dir / "image_ids.json")
    if len(p0_ids_all) < args.n_images:
        raise AssertionError(f"P0 has only {len(p0_ids_all)} IDs; need {args.n_images}")
    cohort = p0_ids_all[:args.n_images]
    layer4_ids = ordered_ids(args.layer4_dir / "image_ids.json")
    layer11_ids = ordered_ids(args.layer11_dir / "image_ids.json")
    assert_same_cohort(cohort, layer4_ids, "layer 4")
    assert_same_cohort(cohort, layer11_ids, "layer 11")

    p0_rows, p0_config = load_layer_rows(args.p0_dir, 8, cohort, p0_source=True)
    layer4_rows, layer4_config = load_layer_rows(args.layer4_dir, 4, cohort, p0_source=False)
    layer11_rows, layer11_config = load_layer_rows(args.layer11_dir, 11, cohort, p0_source=False)
    rows_by_layer = {4: layer4_rows, 8: p0_rows, 11: layer11_rows}
    configs_by_layer = {4: layer4_config, 8: p0_config, 11: layer11_config}
    summaries = [layer_summary(rows_by_layer[layer], layer, args.seed, args.bootstrap_resamples, configs_by_layer[layer]) for layer in LAYER_ORDER]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [row for layer in LAYER_ORDER for row in rows_by_layer[layer]]
    all_rows.sort(key=lambda row: (int(row["layer"]), int(row["stream_position"]), CONDITIONS.index(row["condition"])))
    write_rows(all_rows, output)
    save_json(output / "image_ids.json", cohort)
    summary = {
        "condition_set": "p1_layer_robustness_aggregate",
        "n_images": args.n_images,
        "layers": summaries,
        "cohort_validation": {"status": "passed", "ordered_source_ids_labels_and_split_match": True, "source": "first 256 IDs of the P0 512-image cohort"},
        "claim_boundary": "This is one shared-cohort robustness profile across layers, not three independent replications. Do not combine layer-level tests into an independent p-value.",
        "layer_11_scope_note": "Layer 11 is the final transformer block. With a post-block hook that changes only register positions, there is no later token-mixing block through which the change can reach the final patch positions. The observed zero layer-11 patch effect is therefore structurally uninformative for feature robustness; terminal_layer_locality_check.json records a direct two-image verification.",
        "checkpoint_provenance_note": "dictionary_learning/buffer.py selected the final four sequence positions for registers_only training, while Dinov2WithRegistersEmbeddings inserts registers at [1:5]. These results intervene on actual model register positions and must be interpreted as a correction rather than validation of the older register-SAE claim.",
    }
    save_json(output / "summary.json", summary)
    config = {
        "aggregate_script": str(Path(__file__).relative_to(REPO)),
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "model_name": p0_config["model_name"],
        "model_revision": p0_config["model_revision"],
        "preprocessing": p0_config["preprocessing"],
        "fixed_cohort": "first 256 ordered examples from P0 fixed 512-image validation cohort",
        "conditions": list(CONDITIONS),
        "source_runs": {
            "layer_4": str(args.layer4_dir),
            "layer_8": str(args.p0_dir) + " (filtered to first 256 images and P1 conditions)",
            "layer_11": str(args.layer11_dir),
        },
        "source_elapsed_seconds": {"layer_4": layer4_config.get("elapsed_seconds"), "layer_8_p0_full_run": p0_config.get("elapsed_seconds"), "layer_11": layer11_config.get("elapsed_seconds")},
        "source_hardware": {"layer_4": layer4_config.get("hardware"), "layer_8": p0_config.get("hardware"), "layer_11": layer11_config.get("hardware")},
        "commands": {
            "p0_layer_8": ".venv/bin/python scripts/intervention_controls.py --allow-inferred-legacy-bottom --n-images 512 --device cpu --dtype float32 --streaming-shuffle-buffer 512",
            "p1_layer_4": f".venv/bin/python scripts/intervention_controls.py --condition-set p1 --layer 4 --sae-path {layer4_config['sae_path'].removesuffix('/ae.pt')} --n-images {args.n_images} --device cpu --dtype float32 --streaming-shuffle-buffer 512 --output-dir {args.layer4_dir}",
            "p1_layer_11": f".venv/bin/python scripts/intervention_controls.py --condition-set p1 --layer 11 --sae-path {layer11_config['sae_path'].removesuffix('/ae.pt')} --n-images {args.n_images} --device cpu --dtype float32 --streaming-shuffle-buffer 512 --output-dir {args.layer11_dir}",
            "aggregate": " ".join(sys.argv),
        },
        "hardware_at_aggregation": hardware_metadata(),
        "code_revision": code_revision(),
    }
    save_json(output / "config.json", config)
    render_figure(summaries, output)
    (output / "OUTCOME_MEMO.md").write_text(outcome_memo(summaries), encoding="utf-8")
    (output / "README.md").write_text(
        "P1 corrected-control layer-robustness aggregate. The layer-4, layer-8, and layer-11 results use the same ordered 256-image prefix of the fixed P0 cohort. Layer 8 is filtered from P0; layers 4 and 11 are dedicated P1 runs. See OUTCOME_MEMO.md and summary.json for image-level 10,000-resample bootstrap CIs and raw paired differences. This is a shared-cohort robustness profile, not independent replication evidence. Layer 11 is a terminal post-block hook and therefore cannot alter final patch positions through later token mixing; terminal_layer_locality_check.json documents this directly. Checkpoint token-position provenance is documented in summary.json.\n",
        encoding="utf-8",
    )
    (output / "run.log").write_text("Layer-robustness aggregation completed after exact cohort-ID validation.\n", encoding="utf-8")
    print(f"Wrote P1 aggregate for {args.n_images} images to {output}")


if __name__ == "__main__":
    main()
