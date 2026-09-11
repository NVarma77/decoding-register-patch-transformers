#!/usr/bin/env python3
"""Replay the legacy (h_repl) top-five/bottom-five protocol at the
mistargeted ``registers_only`` checkpoint's own training population.

``intervention_controls.py`` already replays the historical decode-through-the-
SAE protocol (``legacy_top5`` / ``legacy_bottom5``) at TRUE register positions
[1:5) -- the checkpoint's out-of-distribution failure case (FVE 0.007) -- and
that produces the 9.274-point reconstruction confound reported in the paper's
F2 section. The paper's original 48.17%/47.05% top-five/bottom-five pair,
however, was reportedly measured at the population the checkpoint's OWN
(buggy) training selector actually draws from at evaluation time: terminal
patch positions [257:261), where the checkpoint is comparatively well fit
(FVE 0.661, confirmed independently by full_sae_register_audit.py).

This script replays the same legacy protocol -- same checkpoint, same fixed
512-image cohort (seed 20260822), same legacy_top5/legacy_bottom5 selection
rules -- but extracts from and intervenes at [257:261) instead of [1:5). This
converts the top-five/bottom-five near-identity from a number carried from the
original pipeline's report into a number measured under the current protocol
and cohort.

Caveat inherited unchanged from intervention_controls.py: the historical
bottom-five selection algorithm's source implementation is not recoverable.
``legacy_bottom5`` here uses the same explicitly inferred smallest-positive-
mean-active-latent rule used for the true-register replay, not a verified
reproduction of whatever the original notebook did. Results below must be
read with that caveat, exactly as Section 4 already reads the true-register
replay.

Metric note: the paper's fixed endpoint is mean percentage-point cosine drop
over final-layer patch positions [5:261), "excluding CLS and registers". At
true register positions, [1:5) sits outside that range, so the intervened
positions are structurally excluded from their own downstream-effect metric.
Terminal patches [257:261) sit INSIDE [5:261). Keeping the metric definition
fixed (as the paper states it) means the four intervened positions are among
the 256 averaged here, unlike the register-position runs. This is disclosed,
not concealed: it is the faithful replication of the pipeline's own fixed
metric applied at this operating point, not a design change introduced here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from intervention_controls import (
    ActivationHook, output_tensor, encode_decode, contributions, legacy_latents,
    reconstruction_stats, bootstrap_ci, paired_summary, sha256_file, code_revision,
    model_revision, hardware_metadata, preprocessing_metadata, jsonable, save_json,
    select_dataset, make_cfg, hook_description,
)
from utils.utils import load_model, resolve_attr

MODEL_NAME = "facebook/dinov2-with-registers-small"
LAYER = 8
N_TARGET_POSITIONS = 4
TERMINAL_PATCH_START = 257
TERMINAL_PATCH_END = 261
PATCH_START = 5
N_PATCHES = 256
DEFAULT_SAE = REPO / "saes/facebook_dinov2-with-registers-small/enc_res_out_layer_8_top_k_2048_6_1.0_22192860_registers_only/trainer_0"
OUTPUT = REPO / "outputs/experiments/legacy_protocol_terminal_patch_replay"
SEED = 20260822
TOL = 2e-5
CONDITIONS = ["clean", "identity_hook", "reconstruction", "legacy_top5", "legacy_bottom5"]


def terminal_patch_transform(value: torch.Tensor):
    def transform(full: torch.Tensor) -> torch.Tensor:
        if full.ndim != 3 or full.shape[1] < TERMINAL_PATCH_END:
            raise RuntimeError(f"Expected [batch,tokens,width] with {TERMINAL_PATCH_END} positions, got {tuple(full.shape)}")
        out = full.clone()
        replacement = value.to(device=full.device, dtype=full.dtype)
        out[:, TERMINAL_PATCH_START:TERMINAL_PATCH_END, :] = replacement.unsqueeze(0).expand(full.shape[0], -1, -1)
        return out
    return transform


def patch_metric(clean: torch.Tensor, condition: torch.Tensor) -> tuple[float, float]:
    c = clean[:, PATCH_START:PATCH_START + N_PATCHES]
    y = condition[:, PATCH_START:PATCH_START + N_PATCHES]
    cos = F.cosine_similarity(c, y, dim=-1).mean(dim=-1)
    stability = float(cos.item())
    return stability, 100.0 * (1.0 - stability)


def outcome_memo(summary: dict[str, Any], config: dict[str, Any]) -> str:
    contrast = {item["contrast"]: item for item in summary["contrasts"]}
    recon = contrast["reconstruction - clean"]
    lt = contrast["legacy_top5 - reconstruction"]
    lb = contrast["legacy_bottom5 - reconstruction"]
    gap = contrast["legacy_top5 - legacy_bottom5"]
    return f"""# Legacy-protocol replay at the checkpoint's own operating point ([257:261))

Interpretation: same-cohort, same-protocol replay of the mistargeted
`registers_only` checkpoint's historical top-five/bottom-five ablation,
evaluated at terminal patches [257:261) -- the population its training
selector actually drew from -- rather than at true registers [1:5).

Reconstruction alone (no latent removed) produced a {recon['mean']:.3f}%
mean final-patch cosine drop (95% paired bootstrap CI
[{recon['bootstrap_ci95'][0]:.3f}, {recon['bootstrap_ci95'][1]:.3f}];
n={recon['n_images']}). Legacy top-five removal added
{lt['mean']:.3f} points beyond reconstruction (95% CI
[{lt['bootstrap_ci95'][0]:.3f}, {lt['bootstrap_ci95'][1]:.3f}]); legacy
bottom-five (inferred rule) added {lb['mean']:.3f} points beyond
reconstruction (95% CI [{lb['bootstrap_ci95'][0]:.3f}, {lb['bootstrap_ci95'][1]:.3f}]).
Top-five minus bottom-five is {gap['mean']:.3f} points (95% CI
[{gap['bootstrap_ci95'][0]:.3f}, {gap['bootstrap_ci95'][1]:.3f}]).

This uses the same fixed 512-image cohort (seed {config.get('seed')}), the
same checkpoint, and the same legacy_top5 / (inferred) legacy_bottom5
selection rules as the true-register replay in
`outputs/experiments/intervention_controls`. The historical bottom-five
source implementation remains unrecoverable; `legacy_bottom5` here is the
same explicitly inferred smallest-positive-mean-active-latent rule, not a
verified reproduction of the original notebook's algorithm.

Command: `python scripts/legacy_protocol_terminal_patch_replay.py --n-images 512
--device cpu --dtype float32 --streaming-shuffle-buffer 512`.
Elapsed time: {config.get('elapsed_seconds', float('nan')):.1f} seconds.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=512)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--sae-path", type=Path, default=DEFAULT_SAE)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--streaming-shuffle-buffer", type=int, default=512)
    args = ap.parse_args()
    if args.streaming_shuffle_buffer < args.n_images:
        raise ValueError("--streaming-shuffle-buffer must be at least --n-images")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                         handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    start = time.time()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    sae_file = args.sae_path / "ae.pt"
    if not sae_file.exists():
        raise FileNotFoundError(sae_file)
    sae = AutoEncoderTopK.from_pretrained(str(sae_file), device=device)
    sae.eval()
    cfg = make_cfg(str(device), dtype)
    model, _tok, processor = load_model(MODEL_NAME, cfg, dtype=dtype, device=str(device))
    model.eval()
    hook_name = f"enc_res_out_layer_{args.layer}"
    module = resolve_attr(model, f"encoder.layer[{args.layer}]")
    image_records = select_dataset(args.n_images, args.seed, args.split, args.streaming_shuffle_buffer)
    save_json(out / "image_ids.json", [{k: v for k, v in rec.items() if k != "image"} for rec in image_records])
    save_json(out / "config.json", {
        "seed": args.seed, "model_name": MODEL_NAME, "model_revision": model_revision(),
        "layer": args.layer, "hook_name": hook_name, "hook_description": hook_description(args.layer),
        "sae_path": str(sae_file), "sae_sha256": sha256_file(sae_file), "sae_config": str(args.sae_path / "config.json"),
        "split": args.split,
        "cohort_source": f"ILSVRC/imagenet-1k validation streaming shuffle(seed={args.seed}, buffer_size={args.streaming_shuffle_buffer}); identical cohort construction to scripts/intervention_controls.py's default P0 cohort",
        "preprocessing": preprocessing_metadata(), "dtype": args.dtype, "device": str(device),
        "hardware": hardware_metadata(), "code_revision": code_revision(), "command": " ".join(sys.argv),
        "n_requested": args.n_images, "n_selected": len(image_records),
        "token_indexing": {
            "evaluated_positions": "terminal patches [257:261)",
            "why": "population the released registers_only training selector actually drew from (hidden_states[:, -4:, :]); checkpoint is in-distribution here (FVE ~0.661)",
            "true_register_positions_for_contrast": "[1:5)",
            "final_patch_metric": "[5:261), unchanged from intervention_controls.py; this range includes the four intervened terminal-patch positions at this operating point, unlike the true-register replay where they fall outside it",
        },
        "legacy_rule": "top: recovered notebook mean-over-target-position TopK-activation rule; bottom: inferred smallest-positive-mean-active rule, not source-verifiable (see module docstring)",
        "elapsed_seconds_at_write": None,
    })

    transform = processor
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for rec in image_records:
        image = rec["image"]
        inputs = transform(images=image, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=device, dtype=dtype)
        captured: dict[str, torch.Tensor] = {}

        def capture(_m, _i, o):
            captured["x"] = output_tensor(o).detach()
            return o

        cap_handle = module.register_forward_hook(capture)
        with torch.no_grad():
            clean_out = model(pixel_values=pixels)
        cap_handle.remove()
        x = captured["x"][0, TERMINAL_PATCH_START:TERMINAL_PATCH_END].to(dtype=torch.float32)
        with torch.no_grad():
            xhat, f = encode_decode(sae, x)
        clean_repr = clean_out.last_hidden_state.detach()
        mse, rc, fve = reconstruction_stats(x, xhat)
        try:
            legacy_top = legacy_latents(f, "top")
            legacy_bottom = legacy_latents(f, "bottom")
            eligible = True
            reason = ""
        except ValueError as exc:
            eligible = False; reason = str(exc)
            legacy_top = legacy_bottom = []
        dltop = contributions(sae, f, legacy_top)
        dlbottom = contributions(sae, f, legacy_bottom)

        if eligible:
            masked_f_top = f.clone()
            for r, i in legacy_top:
                masked_f_top[r, i] = 0
            masked_decode_error = float((sae.decode(masked_f_top) - (xhat - dltop)).abs().max().item())
            if masked_decode_error > TOL:
                raise AssertionError(f"masked decode mismatch: {masked_decode_error}")
        else:
            masked_decode_error = None

        interventions = {"clean": x, "identity_hook": x, "reconstruction": xhat,
                          "legacy_top5": xhat - dltop, "legacy_bottom5": xhat - dlbottom}
        outputs: dict[str, tuple[float, float]] = {"clean": (1.0, 0.0)}
        identity_error = None
        for condition in CONDITIONS[1:]:
            fn = (lambda z: z) if condition == "identity_hook" else terminal_patch_transform(interventions[condition])
            with ActivationHook(module, fn):
                with torch.no_grad():
                    out_i = model(pixel_values=pixels)
            outputs[condition] = patch_metric(clean_repr, out_i.last_hidden_state.detach())
            if condition == "identity_hook":
                identity_error = float((out_i.last_hidden_state.detach() - clean_repr).abs().max().item())
                if identity_error > TOL:
                    raise AssertionError(f"identity hook mismatch: {identity_error}")

        audits.append({
            "image_id": rec["source_id"], "eligible": eligible, "reason": reason,
            "legacy_top5_latent_ids": sorted({i for _, i in legacy_top}),
            "legacy_bottom5_latent_ids_inferred": sorted({i for _, i in legacy_bottom}),
            "reconstruction_mse": mse, "reconstruction_cosine": rc, "reconstruction_explained_variance": fve,
            "masked_decode_max_abs_error": masked_decode_error, "identity_hook_max_abs_error": identity_error,
        })
        for condition in CONDITIONS:
            stability, drop = outputs[condition]
            rows.append({
                "image_id": rec["source_id"], "stream_position": rec["stream_position"], "label": rec["label"],
                "split": rec["split"], "condition": condition, "eligible": eligible, "eligibility_reason": reason,
                "model_name": MODEL_NAME, "layer": args.layer, "hook_name": hook_name,
                "sae_path": str(sae_file), "cosine_stability": stability, "percentage_drop": drop,
                "reconstruction_mse": mse, "reconstruction_cosine": rc, "reconstruction_explained_variance": fve,
                "seed": args.seed,
            })

    with (out / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as fcsv:
        import csv as _csv
        writer = _csv.DictWriter(fcsv, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out / "per_image_metrics.parquet", index=False)
    except Exception as exc:
        logging.warning("Parquet unavailable: %s", exc)
    save_json(out / "selection_audit.json", audits)

    contrasts = [("reconstruction", "clean"), ("legacy_top5", "reconstruction"),
                 ("legacy_bottom5", "reconstruction"), ("legacy_top5", "legacy_bottom5")]
    summary = {
        "n_rows": len(rows), "n_images": len(image_records),
        "n_eligible": int(sum(bool(a["eligible"]) for a in audits)),
        "excluded_count": int(sum(not a["eligible"] for a in audits)),
        "contrasts": [paired_summary(rows, a, b, args.seed + i * 101) for i, (a, b) in enumerate(contrasts)],
        "legacy_bottom_status": "inferred_rule_only, same as intervention_controls.py",
        "mean_reconstruction_fve_at_terminal_patches": float(np.mean([a["reconstruction_explained_variance"] for a in audits])),
    }
    save_json(out / "summary.json", summary)
    final_config = {**json.loads((out / "config.json").read_text()), "elapsed_seconds": time.time() - start}
    save_json(out / "config.json", final_config)
    (out / "OUTCOME_MEMO.md").write_text(outcome_memo(summary, final_config), encoding="utf-8")
    (out / "README.md").write_text(
        "Legacy-protocol (h_repl) replay at the mistargeted registers_only checkpoint's own "
        "training population, terminal patches [257:261), on the same fixed 512-image cohort "
        "used throughout this paper. See OUTCOME_MEMO.md for the headline numbers and the "
        "module docstring in legacy_protocol_terminal_patch_replay.py for the metric-scope and "
        "bottom-five-rule caveats.\n",
        encoding="utf-8",
    )
    logging.info("Completed in %.1fs; outputs in %s", time.time() - start, out)


if __name__ == "__main__":
    main()
