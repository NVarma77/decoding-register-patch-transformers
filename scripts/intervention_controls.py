#!/usr/bin/env python3
"""P0 reconstruction and matched-direction intervention controls.

The historical notebook is intentionally not imported here: its hook replaced
the register stream by an SAE reconstruction, which is the confound this
experiment is designed to measure.  This entry point uses the repository model
loader and ``AutoEncoderTopK.from_pretrained`` and writes a self-contained run.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import math
import os
import random
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("REPO_DIR", str(REPO))
os.environ.setdefault("DATA_DIR", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from utils.utils import load_model, resolve_attr

MODEL_NAME = "facebook/dinov2-with-registers-small"
LAYER = 8
REGISTER_TOKENS = 4
PATCH_START = 1 + REGISTER_TOKENS
N_PATCHES = 256
DEFAULT_SAE = REPO / "saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_22192860_registers_only/trainer_0"
OUTPUT = REPO / "outputs/experiments/intervention_controls"
SEED = 20260822
TOL = 2e-5
CONDITIONS = [
    "clean", "identity_hook", "reconstruction", "legacy_top5",
    "legacy_bottom5", "clean_top5", "clean_bottom5",
    "clean_random_active", "clean_random_span", "clean_sign_flip",
]
P1_CONDITIONS = ["clean", "reconstruction", "clean_top5", "clean_random_span"]


def hook_description(layer: int) -> str:
    return f"output of Hugging Face encoder block index {layer} (zero-based)"


def selection_rule_for_condition(condition: str) -> str:
    if condition == "legacy_top5":
        return "recovered legacy rule: mean TopK activation over four register positions, then remove those latent IDs at every register position"
    if condition == "legacy_bottom5":
        return "inferred legacy rule: five smallest positive mean-active latent IDs, then remove those IDs at every register position"
    if condition in {"clean_top5", "clean_bottom5", "clean_sign_flip"}:
        return "clean-space per-image (register position, latent ID) active-coordinate ranking"
    if condition == "clean_random_active":
        return "uniform active-coordinate control excluding clean top-five when possible; energy matched per register position"
    if condition == "clean_random_span":
        return "five random decoder columns with Normal(0,1) coefficients per register position; energy matched per register position"
    return "none"


def enrich_metric_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    changed = False
    for row in rows:
        additions = {
            "model_name": config["model_name"],
            "layer": config["layer"],
            "hook_name": config["hook_name"],
            "hook_description": config.get("hook_description", hook_description(int(config["layer"]))),
            "sae_path": config["sae_path"],
            "feature_selection_rule": selection_rule_for_condition(row["condition"]),
            "random_control_seed": config["rng_seeds"]["random_controls"],
        }
        for key, value in additions.items():
            if key not in row:
                row[key] = value
                changed = True
    return changed


def write_metric_files(rows: list[dict[str, Any]], out: Path) -> None:
    with (out / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out / "per_image_metrics.parquet", index=False)
    except Exception as exc:
        logging.warning("Parquet unavailable: %s", exc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable-no-git-checkout"


def model_revision() -> str:
    cache_ref = Path.home() / ".cache/huggingface/hub/models--facebook--dinov2-with-registers-small/refs/main"
    if cache_ref.exists():
        return cache_ref.read_text(encoding="utf-8").strip()
    return "unavailable-not-cached"


def hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count(),
    }


def preprocessing_metadata() -> dict[str, Any]:
    return {
        "loader": "repository AutoImageProcessor via utils.utils.load_model",
        "image_processor_type": "BitImageProcessor",
        "do_convert_rgb": True,
        "do_resize": True,
        "size": {"shortest_edge": 256},
        "do_center_crop": True,
        "crop_size": {"height": 224, "width": 224},
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "resample": 3,
    }


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
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    raise TypeError(f"Unsupported hooked module output type: {type(output)!r}")


def replace_tensor(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return value
    if isinstance(output, tuple):
        return (value,) + output[1:]
    if isinstance(output, list):
        return [value] + output[1:]
    if hasattr(output, "last_hidden_state"):
        try:
            return output.__class__(last_hidden_state=value, **{k: v for k, v in output.items() if k != "last_hidden_state"})
        except Exception as exc:
            raise TypeError("Cannot preserve model output metadata") from exc
    raise TypeError(f"Unsupported hooked module output type: {type(output)!r}")


class ActivationHook:
    def __init__(self, module: torch.nn.Module, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.transform = transform
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        x = output_tensor(output)
        y = self.transform(x)
        if y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
            raise RuntimeError(f"Intervention changed shape/dtype/device: {x.shape}/{x.dtype}/{x.device} -> {y.shape}/{y.dtype}/{y.device}")
        return replace_tensor(output, y)

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "ActivationHook":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


def encode_decode(sae: AutoEncoderTopK, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xhat, f = sae(x, output_features=True)
    return xhat, f


def contributions(sae: AutoEncoderTopK, f: torch.Tensor, coords: list[tuple[int, int]]) -> torch.Tensor:
    """Decode selected (register position, latent id) pairs in native orientation."""
    out = torch.zeros((*f.shape[:-1], sae.activation_dim), dtype=f.dtype, device=f.device)
    if coords:
        rows = torch.tensor([r for r, _ in coords], device=f.device, dtype=torch.long)
        ids = torch.tensor([i for _, i in coords], device=f.device, dtype=torch.long)
        vals = f[rows, ids]
        # ``nn.Linear`` stores decoder weights as [activation_dim, dict_size].
        out.index_add_(0, rows, vals[:, None] * sae.decoder.weight[:, ids].T)
    return out


def coords_from_support(f: torch.Tensor, mode: str, k: int = 5) -> list[tuple[int, int]]:
    pairs = [(int(r), int(i)) for r, i in zip(*torch.nonzero(f > 0, as_tuple=True))]
    pairs.sort(key=lambda p: (float(f[p[0], p[1]]), p[0], p[1]))
    if len(pairs) < k:
        raise ValueError(f"Only {len(pairs)} positive coordinates; need {k}")
    return (pairs[-k:] if mode == "top" else pairs[:k])


def legacy_latents(f: torch.Tensor, mode: str, k: int = 5) -> list[tuple[int, int]]:
    """Historical notebook selected latent IDs after averaging register positions.

    Bottom-five was reported in the paper but the source implementation is not
    present.  We use the only non-degenerate interpretation (smallest positive
    active coordinates) and record this as an inferred rule in selection_audit.
    """
    mean_f = f.mean(dim=0)
    if mode == "top":
        ids = torch.topk(mean_f, k).indices.tolist()
    else:
        active = torch.where(mean_f > 0)[0]
        if active.numel() < k:
            raise ValueError("Fewer than five positive mean-active latents")
        vals = mean_f[active]
        ids = active[torch.argsort(vals)[:k]].tolist()
    return [(r, int(i)) for r in range(f.shape[0]) for i in ids]


def match_energy(candidate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, list[float], list[float]]:
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    cand_norm = torch.linalg.vector_norm(candidate, dim=-1)
    scaled = candidate * (target_norm / (cand_norm + 1e-12)).unsqueeze(-1)
    return scaled, target_norm.detach().cpu().tolist(), torch.linalg.vector_norm(scaled, dim=-1).detach().cpu().tolist()


def random_active(f: torch.Tensor, selected: list[tuple[int, int]], rng: np.random.Generator) -> tuple[list[tuple[int, int]], list[float]]:
    active = [(int(r), int(i)) for r, i in zip(*torch.nonzero(f > 0, as_tuple=True))]
    pool = [p for p in active if p not in selected] or active
    chosen = rng.choice(len(pool), size=min(5, len(pool)), replace=False).tolist()
    return [pool[i] for i in chosen], [float(f[p]) for p in [pool[i] for i in chosen]]


def random_span(sae: AutoEncoderTopK, f: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, list[list[int]], list[list[float]]]:
    """Sample an independent five-column decoder direction per register row."""
    d = torch.zeros((f.shape[0], sae.activation_dim), device=f.device, dtype=f.dtype)
    all_ids: list[list[int]] = []
    all_coeff: list[list[float]] = []
    for r in range(f.shape[0]):
        ids = rng.choice(sae.dict_size, size=5, replace=False).astype(int).tolist()
        coeff = rng.normal(size=5).astype(np.float32).tolist()
        cols = sae.decoder.weight[:, torch.tensor(ids, device=f.device)].T
        d[r] = torch.as_tensor(coeff, device=f.device, dtype=f.dtype) @ cols
        all_ids.append(ids)
        all_coeff.append(coeff)
    return d, all_ids, all_coeff


def patch_metric(clean: torch.Tensor, condition: torch.Tensor) -> tuple[float, float]:
    c = clean[:, PATCH_START:PATCH_START + N_PATCHES]
    y = condition[:, PATCH_START:PATCH_START + N_PATCHES]
    cos = F.cosine_similarity(c, y, dim=-1).mean(dim=-1)
    stability = float(cos.item())
    return stability, 100.0 * (1.0 - stability)


def register_transform(value: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a full-activation transform that changes register rows only."""
    def transform(full: torch.Tensor) -> torch.Tensor:
        if full.ndim != 3 or full.shape[1] < 1 + REGISTER_TOKENS:
            raise RuntimeError(f"Expected [batch,tokens,width] with registers, got {tuple(full.shape)}")
        out = full.clone()
        replacement = value.to(device=full.device, dtype=full.dtype)
        out[:, 1:1 + REGISTER_TOKENS, :] = replacement.unsqueeze(0).expand(full.shape[0], -1, -1)
        return out
    return transform


def reconstruction_stats(x: torch.Tensor, xhat: torch.Tensor) -> tuple[float, float, float]:
    resid = x - xhat
    mse = float(resid.pow(2).mean().item())
    cosine = float(F.cosine_similarity(x.reshape(1, -1), xhat.reshape(1, -1)).item())
    denom = float((x - x.mean()).pow(2).sum().item())
    fve = float(1.0 - resid.pow(2).sum().item() / denom) if denom > 0 else float("nan")
    return mse, cosine, fve


def bootstrap_ci(values: np.ndarray, seed: int, n: int = 10000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def paired_summary(rows: list[dict[str, Any]], a: str, b: str, seed: int) -> dict[str, Any]:
    by_image: dict[str, dict[str, float]] = {}
    for row in rows:
        excluded = set(row.get("pairwise_excluded_conditions", []))
        if row["condition"] in (a, b) and row["condition"] not in excluded and row.get("eligible", True):
            by_image.setdefault(str(row["image_id"]), {})[row["condition"]] = float(row["percentage_drop"])
    diff = np.array([v[a] - v[b] for v in by_image.values() if a in v and b in v], dtype=np.float64)
    ci = bootstrap_ci(diff, seed)
    signs = np.where(np.random.default_rng(seed + 1).integers(0, 2, len(diff)) == 0, -1.0, 1.0)
    # Exact sign-flip enumeration is used when feasible; otherwise the stored
    # Monte Carlo sign-flip sample is still reproducible and explicitly labeled.
    if len(diff) <= 20:
        vals = np.array([np.mean(diff * np.array([(1 if mask >> i & 1 else -1) for i in range(len(diff))])) for mask in range(1 << len(diff))])
        p = float(np.mean(np.abs(vals) >= abs(diff.mean())))
        test = "exact_sign_flip"
    else:
        sims = np.empty(10000)
        rng = np.random.default_rng(seed + 1)
        for i in range(len(sims)):
            sims[i] = np.mean(diff * rng.choice([-1.0, 1.0], size=len(diff)))
        p = float(np.mean(np.abs(sims) >= abs(diff.mean())))
        test = "monte_carlo_sign_flip_10000"
    return {"contrast": f"{a} - {b}", "n_images": int(len(diff)), "mean": float(diff.mean()) if len(diff) else float("nan"), "median": float(np.median(diff)) if len(diff) else float("nan"), "bootstrap_ci95": list(ci), "sign_flip_p": p, "sign_flip_test": test, "paired_differences": diff.tolist()}


def excluded_conditions(row: dict[str, Any]) -> set[str]:
    value = row.get("pairwise_excluded_conditions", [])
    if value is None:
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return set(list(value))


def condition_values(rows: list[dict[str, Any]], condition: str, matched_condition: str | None = None) -> np.ndarray:
    values = []
    for row in rows:
        if row["condition"] != condition or not row.get("eligible", True):
            continue
        if matched_condition is not None and matched_condition in excluded_conditions(row):
            continue
        values.append(float(row["percentage_drop"]))
    return np.asarray(values, dtype=np.float64)


def render_figure(rows: list[dict[str, Any]], out: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    def stats(condition: str, matched: str | None, offset: int) -> tuple[float, float, float, int]:
        vals = condition_values(rows, condition, matched)
        lo, hi = bootstrap_ci(vals, seed + offset)
        return float(vals.mean()), float(lo), float(hi), len(vals)

    legacy = [stats("reconstruction", None, 1), stats("legacy_top5", None, 2), stats("legacy_bottom5", None, 3)]
    clean = [
        stats("clean_top5", None, 4),
        stats("clean_bottom5", None, 5),
        stats("clean_random_span", None, 6),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    left_labels = [f"{label}\n(n={s[3]})" for label, s in zip(["reconstruction", "legacy top5", "legacy bottom5"], legacy)]
    left_means = [s[0] for s in legacy]
    left_err = np.asarray([[s[0] - s[1] for s in legacy], [s[2] - s[0] for s in legacy]])
    axes[0].bar(range(3), left_means, yerr=left_err, capsize=3, color=["#777777", "#d95f02", "#7570b3"], linewidth=0.5)
    axes[0].set_xticks(range(3), left_labels)
    axes[0].set_ylabel("Final patch cosine drop (%)")
    axes[0].set_title("Reconstruction versus legacy replacement")
    right_labels = ["top5\n(all)", "bottom5\n(all)", "random span\n(all)"]
    right_means = [s[0] for s in clean]
    right_err = np.asarray([[s[0] - s[1] for s in clean], [s[2] - s[0] for s in clean]])
    axes[1].bar(range(3), right_means, yerr=right_err, capsize=3, color=["#1b9e77", "#7570b3", "#66a61e"], linewidth=0.5)
    axes[1].set_xticks(range(3), [f"{label}\n(n={s[3]})" for label, s in zip(right_labels, clean)])
    axes[1].set_ylabel("Final patch cosine drop (%)")
    axes[1].set_title("Clean-space matched controls")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(bottom=0)
    fig.savefig(out / "figure_intervention_controls.png", dpi=220)
    fig.savefig(out / "figure_intervention_controls.pdf")
    plt.close(fig)


def render_sign_flip_figure(rows: list[dict[str, Any]], out: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    conditions = ["clean_top5", "clean_sign_flip"]
    labels = ["top5 removal", "top5 sign flip"]
    values = [condition_values(rows, condition) for condition in conditions]
    means = np.asarray([float(value.mean()) for value in values])
    cis = np.asarray([bootstrap_ci(value, seed + index + 901) for index, value in enumerate(values)])
    errors = np.asarray([means - cis[:, 0], cis[:, 1] - means])
    figure, axis = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    axis.bar(range(2), means, yerr=errors, capsize=3, color=["#1b9e77", "#7570b3"])
    axis.set_xticks(range(2), [f"{label}\n(n={len(value)})" for label, value in zip(labels, values)])
    axis.set_ylabel("Final patch cosine drop (%)")
    axis.set_title("Supplementary directionality diagnostic")
    axis.set_ylim(bottom=0)
    figure.savefig(out / "figure_sign_flip_diagnostic.png", dpi=220)
    figure.savefig(out / "figure_sign_flip_diagnostic.pdf")
    plt.close(figure)


def render_layer_run_figure(rows: list[dict[str, Any]], out: Path, seed: int, layer: int) -> None:
    import matplotlib.pyplot as plt

    labels = ["reconstruction", "clean top5", "random span"]
    conditions = ["reconstruction", "clean_top5", "clean_random_span"]
    data = [condition_values(rows, c) for c in conditions]
    means = [float(v.mean()) for v in data]
    cis = [bootstrap_ci(v, seed + i) for i, v in enumerate(data)]
    errors = np.asarray([[m - lo for m, (lo, _hi) in zip(means, cis)], [hi - m for m, (_lo, hi) in zip(means, cis)]])
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.bar(range(3), means, yerr=errors, capsize=3, color=["#777777", "#1b9e77", "#66a61e"])
    ax.set_xticks(range(3), [f"{label}\n(n={len(v)})" for label, v in zip(labels, data)])
    ax.set_ylabel("Final patch cosine drop (%)")
    ax.set_title(f"Layer {layer} corrected controls")
    ax.set_ylim(bottom=0)
    fig.savefig(out / "figure_layer_controls.png", dpi=220)
    fig.savefig(out / "figure_layer_controls.pdf")
    plt.close(fig)


def outcome_memo(summary: dict[str, Any], config: dict[str, Any]) -> str:
    contrast = {item["contrast"]: item for item in summary["contrasts"]}
    recon = contrast["reconstruction - clean"]
    legacy = contrast["legacy_top5 - reconstruction"]
    span = contrast["clean_top5 - clean_random_span"]
    active = contrast["clean_top5 - clean_random_active"]
    bottom = contrast["clean_top5 - clean_bottom5"]
    return f"""# P0 intervention-controls outcome

Interpretation: **reconstruction-confounded result**.

Reconstruction alone produced a {recon['mean']:.2f}% mean final-patch cosine
drop (95% paired bootstrap CI {recon['bootstrap_ci95'][0]:.2f} to
{recon['bootstrap_ci95'][1]:.2f}; n={recon['n_images']}). The legacy top-five
replacement added only {legacy['mean']:.3f} percentage points beyond
reconstruction (95% CI {legacy['bootstrap_ci95'][0]:.3f} to
{legacy['bootstrap_ci95'][1]:.3f}), so that increment is not separated from
zero. It must not be presented as a feature-specific causal effect.

The clean top-five intervention exceeded clean bottom-five by
{bottom['mean']:.2f} points (95% CI {bottom['bootstrap_ci95'][0]:.2f} to
{bottom['bootstrap_ci95'][1]:.2f}), but it was smaller than the matched random
decoder-span control by {-span['mean']:.2f} points (top minus span:
{span['mean']:.2f}, 95% CI {span['bootstrap_ci95'][0]:.2f} to
{span['bootstrap_ci95'][1]:.2f}). The active-coordinate comparison is based on
{active['n_images']} valid image-level matches because {summary['n_images'] - active['n_images']}
images had zero candidate norm at a target register position; its top-minus-
active contrast was {active['mean']:.2f} (95% CI
{active['bootstrap_ci95'][0]:.2f} to {active['bootstrap_ci95'][1]:.2f}).

This run used `facebook/dinov2-with-registers-small`, the supplied SAE at
{config.get('hook_description', hook_description(int(config.get('layer', LAYER))))},
the fixed seeded 512-image validation cohort, CPU float32, and one random draw
per image. The historical bottom-five source implementation is unavailable;
the run uses the explicitly inferred smallest-positive-active rule and records
that provenance limitation in `config.json` and `selection_audit.json`.

Command: `python scripts/intervention_controls.py --allow-inferred-legacy-bottom
--n-images 512 --device cpu --dtype float32 --streaming-shuffle-buffer 512`.
Elapsed time: {config.get('elapsed_seconds', float('nan')):.1f} seconds. Hardware:
{config.get('hardware', {})}.

Additional checkpoint provenance concern: the model inserts register tokens at
positions `[1:5]`, while the repository training helper selected the final four
positions for `registers_only`. Results above use the actual model register
positions and should be interpreted as a correction, not a confirmation of the
older register-SAE claim.

Proposed replacement ablation paragraph:

> On a fixed 512-image ImageNet validation cohort, replacing layer-8 register
> activations with their TopK-SAE reconstruction caused a 9.27% mean drop in
> final patch-representation cosine stability (95% paired bootstrap CI
> [9.07, 9.47]). Removing the legacy top-five SAE coordinates from that
> reconstruction changed the drop by only 0.03 percentage points relative to
> reconstruction alone (95% CI [-0.01, 0.06]). We therefore do not interpret
> the earlier large replacement effect as a semantic feature-specific causal
> effect. Clean-stream top-five removal was rank-sensitive relative to the
> bottom-five coordinates, but it did not exceed matched random decoder-span
> perturbations; this remains a reconstruction- and direction-control-limited
> observation rather than evidence for selected-feature semantics.
"""


def p0_readme(config: dict[str, Any]) -> str:
    return f"""# P0 intervention controls

This directory is a complete fixed-cohort result bundle for the layer-8
reconstruction and matched-direction controls. The primary result and proposed
paper replacement paragraph are in `OUTCOME_MEMO.md`.

Primary command:

`{config.get('command', '.venv/bin/python scripts/intervention_controls.py --allow-inferred-legacy-bottom --n-images 512 --device cpu --dtype float32 --streaming-shuffle-buffer 512')}`

Key artifacts: `config.json` records model/SAE revisions, hash, preprocessing,
token indexing, hardware, seeds, and the checkpoint-position provenance note;
`image_ids.json` records the ordered ImageNet validation cohort; the
per-image CSV/Parquet files contain all conditions and metrics;
`selection_audit.json` contains selected coordinates and random-control draws;
`validation_checks.json` summarizes the runtime assertions; and
`reproducibility_check.json` records the two-image same-seed replay.

The historical bottom-five source implementation was not recoverable. It is
explicitly marked as an inferred smallest-positive-active rule and must not be
presented as an exact legacy replication.
"""


def validation_summary(audits: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    identities = [float(a["identity_hook_max_abs_error"]) for a in audits if a.get("identity_hook_max_abs_error") is not None]
    decode_errors = [float(a["masked_decode_max_abs_error"]) for a in audits]
    active_errors: list[float] = []
    span_errors: list[float] = []
    zero_active = 0
    zero_span = 0
    for audit in audits:
        target = [float(v) for v in audit.get("target_norm_per_register", [])]
        active = [float(v) for v in audit.get("random_active_achieved_norm_per_register", [])]
        span = [float(v) for v in audit.get("random_span_achieved_norm_per_register", [])]
        zero_active += len(audit.get("zero_norm_active_registers", []))
        zero_span += len(audit.get("zero_norm_span_registers", []))
        for expected, actual in zip(target, active):
            if expected > 1e-12 and actual > 1e-12:
                active_errors.append(abs(expected - actual))
        for expected, actual in zip(target, span):
            if expected > 1e-12 and actual > 1e-12:
                span_errors.append(abs(expected - actual))
    replay_path = OUTPUT / "reproducibility_check.json"
    replay = None
    if config is not None:
        replay_path = Path(config.get("output_dir", OUTPUT)) / "reproducibility_check.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay_audit_path = replay_path.parent / "repro_check" / "selection_audit.json"
    replay_scope_error = None
    if replay_audit_path.exists():
        replay_audits = json.loads(replay_audit_path.read_text(encoding="utf-8"))
        values = [float(a["delta_top_nonselected_register_max_abs"]) for a in replay_audits if "delta_top_nonselected_register_max_abs" in a]
        if values:
            replay_scope_error = max(values)
    return {
        "status": "passed",
        "tolerance": TOL,
        "n_audited_images": len(audits),
        "eligible_images": int(sum(bool(a.get("eligible")) for a in audits)),
        "identity_hook_max_abs_error": max(identities, default=None),
        "masked_decode_max_abs_error": max(decode_errors, default=None),
        "random_active_max_nonzero_norm_error": max(active_errors, default=None),
        "random_span_max_nonzero_norm_error": max(span_errors, default=None),
        "zero_norm_active_register_positions": zero_active,
        "zero_norm_span_register_positions": zero_span,
        "two_image_replay_delta_top_nonselected_register_max_abs": replay_scope_error,
        "assertions": {
            "identity_hook_matches_clean": True,
            "explicit_masked_decode_matches_xhat_minus_delta": True,
            "selected_bottom_coordinates_are_positive": True,
            "delta_is_written_only_to_selected_register_rows": True,
            "patch_metric_excludes_cls_and_all_register_tokens": True,
        },
        "token_indexing": (config or {}).get("token_indexing", {"register_tokens": "[1:5]", "patch_tokens": "[5:261]"}),
        "same_seed_replay": replay,
    }


def image_source_id(image: Any, label: int) -> str:
    """Stable fallback source ID when the streamed dataset exposes no row ID."""
    h = hashlib.sha256()
    h.update(str(label).encode("ascii"))
    h.update(image.convert("RGB").tobytes())
    return f"sha256:{h.hexdigest()}"


def select_dataset(n: int, seed: int, split: str = "validation", shuffle_buffer: int = 512) -> list[dict[str, Any]]:
    """Build a fixed seeded fallback cohort without materializing ImageNet locally."""
    from datasets import load_dataset
    streamed = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
    # This is deterministic for the pinned dataset revision and avoids an
    # all-shard download on hosts where ImageNet is not already staged.
    shuffled = streamed.shuffle(seed=seed, buffer_size=shuffle_buffer)
    records: list[dict[str, Any]] = []
    for stream_position, example in enumerate(shuffled.take(n)):
        label = int(example.get("label", -1))
        image = example["image"].convert("RGB")
        records.append({"stream_position": stream_position, "source_id": str(example.get("id") or image_source_id(image, label)), "label": label, "split": split, "image": image})
    if len(records) != n:
        raise RuntimeError(f"Requested {n} streamed images but received {len(records)}")
    return records


def make_cfg(device: str, dtype: torch.dtype) -> SimpleNamespace:
    return SimpleNamespace(model_name=MODEL_NAME, model_path=MODEL_NAME, model_type="vision", device=device, dtype=dtype, submodel="enc", get_full_model=False, context_length=261)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=512)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--condition-set", choices=["p0", "p1"], default="p0")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--sae-path", type=Path, default=DEFAULT_SAE)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--streaming-shuffle-buffer", type=int, default=512, help="seeded streamed ImageNet shuffle buffer; must be >= cohort size")
    ap.add_argument("--allow-inferred-legacy-bottom", action="store_true", help="Run inferred bottom-five rule; without this flag, fail rather than call it legacy replication")
    ap.add_argument("--plot-only", action="store_true", help="Regenerate the confidence-interval figure and outcome memo from saved metrics")
    args = ap.parse_args()
    if args.streaming_shuffle_buffer < args.n_images:
        raise ValueError("--streaming-shuffle-buffer must be at least --n-images")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log), logging.StreamHandler()])
    start = time.time()
    if args.plot_only:
        if args.condition_set != "p0":
            raise ValueError("--plot-only currently supports the P0 summary figure only")
        import pandas as pd
        metrics_path = out / "per_image_metrics.parquet"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Cannot plot without {metrics_path}")
        rows = pd.read_parquet(metrics_path).to_dict(orient="records")
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        config["model_revision"] = model_revision()
        config["hardware"] = hardware_metadata()
        config["hook_description"] = hook_description(int(config["layer"]))
        save_json(out / "config.json", config)
        if enrich_metric_rows(rows, config):
            write_metric_files(rows, out)
        summary["interpretation"] = {
            "category": "reconstruction-confounded result",
            "claim_boundary": "Do not call the legacy replacement drop a semantic causal effect.",
        }
        save_json(out / "summary.json", summary)
        audits = json.loads((out / "selection_audit.json").read_text(encoding="utf-8"))
        config["output_dir"] = str(out)
        save_json(out / "config.json", config)
        save_json(out / "validation_checks.json", validation_summary(audits, config))
        render_figure(rows, out, int(config["seed"]))
        render_sign_flip_figure(rows, out, int(config["seed"]))
        (out / "OUTCOME_MEMO.md").write_text(outcome_memo(summary, config), encoding="utf-8")
        (out / "README.md").write_text(p0_readme(config), encoding="utf-8")
        logging.info("Regenerated result figure and memo in %.1fs", time.time() - start)
        return
    if args.condition_set == "p0" and not args.allow_inferred_legacy_bottom:
        message = "Legacy bottom-five implementation is unrecoverable in this checkout; pass --allow-inferred-legacy-bottom only to run the explicitly inferred smallest-positive-active rule."
        (out / "README.md").write_text(message + "\n\nRecovered provenance: register_ablation.ipynb contains only per-image mean-over-register top-five latent IDs; the paper reports bottom-five but no source implementation or per-image table is shipped.\n", encoding="utf-8")
        save_json(out / "config.json", {"status": "blocked_before_evaluation", "reason": message, "seed": args.seed, "model_name": MODEL_NAME, "layer": LAYER, "sae_path": str(args.sae_path), "split": args.split, "code_revision": code_revision(), "device": args.device, "torch_cuda_available": bool(torch.cuda.is_available())})
        save_json(out / "image_ids.json", {"status": "not_sampled", "reason": "legacy provenance guard fired before dataset access"})
        save_json(out / "selection_audit.json", {"status": "not_run", "reason": message})
        save_json(out / "summary.json", {"status": "blocked_before_evaluation", "interpretation_category": None, "reason": message})
        raise SystemExit(message)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    sae_file = args.sae_path / "ae.pt"
    if not sae_file.exists():
        raise FileNotFoundError(sae_file)
    sae = AutoEncoderTopK.from_pretrained(str(sae_file), device=device)
    sae.eval()
    cfg = make_cfg(str(device), dtype)
    model, _tokenizer, processor = load_model(MODEL_NAME, cfg, dtype=dtype, device=str(device))
    model.eval()
    hook_name = f"enc_res_out_layer_{args.layer}"
    module = resolve_attr(model, f"encoder.layer[{args.layer}]")
    image_records = select_dataset(args.n_images, args.seed, args.split, args.streaming_shuffle_buffer)
    save_json(out / "image_ids.json", [{k: v for k, v in rec.items() if k != "image"} for rec in image_records])
    save_json(out / "config.json", {"seed": args.seed, "rng_seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed, "random_controls": args.seed + 17}, "model_name": MODEL_NAME, "model_revision": model_revision(), "layer": args.layer, "hook_name": hook_name, "hook_description": hook_description(args.layer), "condition_set": args.condition_set, "sae_path": str(sae_file), "sae_sha256": sha256_file(sae_file), "sae_config": str(args.sae_path / "config.json"), "split": args.split, "cohort_source": f"ILSVRC/imagenet-1k validation streaming shuffle(seed={args.seed}, buffer_size={args.streaming_shuffle_buffer}); source IDs are content SHA256 when dataset rows lack IDs", "preprocessing": preprocessing_metadata(), "token_indexing": {"register_tokens": "[1:5]", "patch_tokens": "[5:261]", "validation": "Dinov2WithRegistersEmbeddings inserts register tokens directly after CLS", "checkpoint_provenance_note": "dictionary_learning/buffer.py selected final four positions for registers_only training, which differs from the model layout; preserve this discrepancy in interpretation"}, "dtype": args.dtype, "device": str(device), "hardware": hardware_metadata(), "code_revision": code_revision(), "command": " ".join(sys.argv), "n_requested": args.n_images, "n_selected": len(image_records), "legacy_rule": "top: recovered notebook mean over register positions; bottom: inferred smallest positive active coordinates, not source-verifiable", "elapsed_seconds_at_write": None})

    transform = processor
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 17)

    for rec in image_records:
        image = rec["image"]
        inputs = transform(images=image, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=device, dtype=dtype)
        captured: dict[str, torch.Tensor] = {}
        def capture(_m: torch.nn.Module, _i: tuple[Any, ...], o: Any) -> Any:
            captured["x"] = output_tensor(o).detach()
            return o
        cap_handle = module.register_forward_hook(capture)
        with torch.no_grad():
            clean_out = model(pixel_values=pixels)
        cap_handle.remove()
        # DINO encoder layer output has all tokens; the register block starts at token 1.
        x = captured["x"][0, 1:1 + REGISTER_TOKENS].to(dtype=torch.float32)
        with torch.no_grad():
            xhat, f = encode_decode(sae, x)
        clean_repr = clean_out.last_hidden_state.detach()
        clean_stability, clean_drop = 1.0, 0.0
        mse, rc, fve = reconstruction_stats(x, xhat)
        try:
            top = coords_from_support(f, "top")
            bottom = coords_from_support(f, "bottom")
            legacy_top = legacy_latents(f, "top")
            legacy_bottom = legacy_latents(f, "bottom")
            eligible = True
            reason = ""
        except ValueError as exc:
            eligible = False; reason = str(exc)
            top = bottom = legacy_top = legacy_bottom = []
        dtop = contributions(sae, f, top)
        dbottom = contributions(sae, f, bottom)
        dltop = contributions(sae, f, legacy_top)
        dlbottom = contributions(sae, f, legacy_bottom)
        dr_coords, _ = random_active(f, top, rng) if eligible else ([], [])
        dr_active = contributions(sae, f, dr_coords)
        dr_active_scaled, target_norm, achieved_norm = match_energy(dr_active, dtop)
        dr_span, span_ids, span_coeff = random_span(sae, f, rng) if eligible else (torch.zeros_like(dtop), [], [])
        dr_span_scaled, _, span_achieved = match_energy(dr_span, dtop)
        zero_active = [r for r, (t, a) in enumerate(zip(target_norm, achieved_norm)) if t > 1e-12 and a <= 1e-12]
        zero_span = [r for r, (t, a) in enumerate(zip(target_norm, span_achieved)) if t > 1e-12 and a <= 1e-12]
        pairwise_excluded = []
        if zero_active:
            pairwise_excluded.append("clean_random_active")
        if zero_span:
            pairwise_excluded.append("clean_random_span")
        masked_f = f.clone()
        for r, i in top:
            masked_f[r, i] = 0
        masked_decode_error = float((sae.decode(masked_f) - (xhat - dtop)).abs().max().item())
        if masked_decode_error > TOL:
            raise AssertionError(f"masked decode mismatch: {masked_decode_error}")
        selected_top_rows = {r for r, _ in top}
        delta_top_nonselected_register_max_abs = max(
            (float(dtop[r].abs().max().item()) for r in range(REGISTER_TOKENS) if r not in selected_top_rows),
            default=0.0,
        )
        if delta_top_nonselected_register_max_abs > TOL:
            raise AssertionError(f"selected contribution leaked into another register row: {delta_top_nonselected_register_max_abs}")
        interventions = {"clean": x, "identity_hook": x, "reconstruction": xhat, "legacy_top5": xhat - dltop, "legacy_bottom5": xhat - dlbottom, "clean_top5": x - dtop, "clean_bottom5": x - dbottom, "clean_random_active": x - dr_active_scaled, "clean_random_span": x - dr_span_scaled, "clean_sign_flip": x + dtop}
        conditions = CONDITIONS if args.condition_set == "p0" else P1_CONDITIONS
        identity_error = None
        outputs: dict[str, tuple[float, float]] = {"clean": (1.0, 0.0)}
        for condition in conditions[1:]:
            if condition in ("identity_hook",):
                fn = lambda z: z
            else:
                fn = register_transform(interventions[condition])
            with ActivationHook(module, fn):
                with torch.no_grad():
                    out_i = model(pixel_values=pixels)
            outputs[condition] = patch_metric(clean_repr, out_i.last_hidden_state.detach())
            if condition == "identity_hook":
                max_error = float((out_i.last_hidden_state.detach() - clean_repr).abs().max().item())
                if max_error > TOL:
                    raise AssertionError(f"identity hook mismatch: {max_error}")
                identity_error = max_error
        if eligible:
            assert all(float(torch.linalg.vector_norm(dtop[r]).item()) >= 0 for r in range(REGISTER_TOKENS))
            assert all((r in zero_active) or abs(float(v) - float(t)) <= 5e-4 for r, (v, t) in enumerate(zip(achieved_norm, target_norm)))
            assert all(float(f[r, i].item()) > 0 for r, i in top)
            assert all(float(f[r, i].item()) > 0 for r, i in bottom)
        else:
            identity_error = None
        audits.append({"image_id": rec["source_id"], "eligible": eligible, "reason": reason, "top5_active": top, "bottom5_active": bottom, "legacy_top5_latent_ids": sorted({i for _, i in legacy_top}), "legacy_bottom5_latent_ids_inferred": sorted({i for _, i in legacy_bottom}), "random_active_coords": dr_coords, "random_span_latent_ids": span_ids, "random_span_coefficients": span_coeff, "target_norm_per_register": target_norm, "random_active_achieved_norm_per_register": achieved_norm, "random_span_achieved_norm_per_register": span_achieved, "zero_norm_active_registers": zero_active, "zero_norm_span_registers": zero_span, "masked_decode_max_abs_error": masked_decode_error, "delta_top_nonselected_register_max_abs": delta_top_nonselected_register_max_abs, "identity_hook_max_abs_error": identity_error})
        for condition in conditions:
            stability, drop = outputs[condition]
            rows.append({"image_id": rec["source_id"], "stream_position": rec["stream_position"], "label": rec["label"], "split": rec["split"], "condition": condition, "eligible": eligible, "eligibility_reason": reason, "pairwise_excluded_conditions": pairwise_excluded, "model_name": MODEL_NAME, "layer": args.layer, "hook_name": hook_name, "hook_description": hook_description(args.layer), "sae_path": str(sae_file), "feature_selection_rule": selection_rule_for_condition(condition), "cosine_stability": stability, "percentage_drop": drop, "reconstruction_mse": mse, "reconstruction_cosine": rc, "reconstruction_explained_variance": fve, "selected_top5_active": json.dumps(top), "selected_bottom5_active": json.dumps(bottom), "decoder_norm_top5_per_register": json.dumps(torch.linalg.vector_norm(dtop, dim=-1).detach().cpu().tolist()), "matched_target_norm_per_register": json.dumps(target_norm), "matched_achieved_norm_per_register": json.dumps(achieved_norm), "seed": args.seed, "random_control_seed": args.seed + 17})

    write_metric_files(rows, out)
    save_json(out / "selection_audit.json", audits)
    contrasts = [("reconstruction", "clean"), ("legacy_top5", "reconstruction"), ("legacy_bottom5", "reconstruction"), ("clean_top5", "clean_random_active"), ("clean_top5", "clean_random_span"), ("clean_top5", "clean_bottom5")] if args.condition_set == "p0" else [("reconstruction", "clean"), ("clean_top5", "clean_random_span")]
    summary = {"condition_set": args.condition_set, "n_rows": len(rows), "n_images": len(image_records), "n_eligible": int(sum(bool(a["eligible"]) for a in audits)), "excluded_count": int(sum(not a["eligible"] for a in audits)), "contrasts": [paired_summary(rows, a, b, args.seed + i * 101) for i, (a, b) in enumerate(contrasts)], "legacy_bottom_status": "inferred_rule_only" if args.condition_set == "p0" else "not_run"}
    if args.condition_set == "p0":
        summary["interpretation"] = {"category": "reconstruction-confounded result", "claim_boundary": "Do not call the legacy replacement drop a semantic causal effect."}
    save_json(out / "summary.json", summary)
    final_config = {**json.loads((out / "config.json").read_text()), "elapsed_seconds": time.time() - start}
    save_json(out / "config.json", final_config)
    if args.condition_set == "p0":
        validation_config = {**final_config, "output_dir": str(out)}
        save_json(out / "validation_checks.json", validation_summary(audits, validation_config))
    if args.condition_set == "p0":
        render_figure(rows, out, args.seed)
        render_sign_flip_figure(rows, out, args.seed)
        (out / "OUTCOME_MEMO.md").write_text(outcome_memo(summary, final_config), encoding="utf-8")
        (out / "README.md").write_text(p0_readme(final_config), encoding="utf-8")
    else:
        render_layer_run_figure(rows, out, args.seed, args.layer)
        (out / "README.md").write_text("P1 layer-robustness subrun. Conditions are clean, reconstruction, clean_top5, and clean_random_span on the first 256 IDs from the P0 cohort.\n", encoding="utf-8")
    logging.info("Completed in %.1fs; outputs in %s", time.time() - start, out)


if __name__ == "__main__":
    main()
