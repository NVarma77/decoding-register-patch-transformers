"""Resample ablation: an exactly on-support null, unscaled.

Referee objection this answers.  The support-matched null of
`support_matched_null.py` displaces the intervened register toward another
image's register at the same slot and then *rescales to the treatment's
energy*.  Registers at a given slot are near-constant across images, so that
difference vector is small and the rescaling amplifies it by a large factor:
the direction is one the model encounters, the magnitude along it is not.  The
paper's 50/50 decomposition of the gap into on- and off-support parts is only
valid if the null is genuinely on support.

We therefore run the null that needs no rescaling argument: replace the
intervened register wholesale with another image's register at the same slot.
That state is one the model demonstrably produces.  We report the
amplification factor the rescaled version applies, so the two are comparable,
and add a plain zero-ablation reference (no SAE) to show what the treatment
costs when described as what it physically is.

Conditions, all at the single slot the top-five selection acts on:
  clean_top5      subtract the selected decoder contribution   (the treatment)
  swap_slot       replace with image j's register, unscaled    (on support)
  rescaled_slot   displace toward image j, rescaled to treatment energy
  zero_slot       set the register to zero, no SAE involved
  random_span     energy-matched random decoder direction      (anchor)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.full_sae_register_audit import (  # noqa: E402
    MODEL_NAME, N_PATCHES, PATCH_START, REGISTER_END, REGISTER_START, SEED,
    coordinates_from_positive_support, deterministic_rng, encode_decode, make_cfg,
    match_energy, output_tensor, random_decoder_span, selected_contribution,
)
from scripts.subspace_matched_control import paired_bootstrap  # noqa: E402
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from utils.utils import load_model  # noqa: E402


def tail_from_layer(model, hidden, layer):
    for i in range(layer + 1, len(model.encoder.layer)):
        hidden = model.encoder.layer[i](hidden, None, False)[0]
    return model.layernorm(hidden)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-ids", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--cohort-cache", type=Path, required=True)
    ap.add_argument("--allow-missing", type=int, default=2)
    ap.add_argument("--sae", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n-images", type=int, default=511)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output_dir / "run.log"),
                                  logging.StreamHandler()], force=True)
    start = time.time()
    torch.manual_seed(SEED)
    dev = torch.device(args.device); dt = torch.float32

    from scripts.cohort_local import load_cohort
    records = load_cohort(args.image_ids, args.workdir, args.cohort_cache,
                          args.allow_missing)[: args.n_images]
    model, _t, proc = load_model(args.model_name, make_cfg(str(dev), dt, args.model_name),
                                 dtype=dt, device=str(dev))
    model.eval()
    sae = AutoEncoderTopK.from_pretrained(str(args.sae / "ae.pt"), device=str(dev))
    logging.info("n=%d layer=%d draws=%d", len(records), args.layer, args.draws)

    # Pass 1: cache every image's registers, so a draw can be another image's.
    regs = []
    for rec in records:
        px = proc(images=rec["image"], return_tensors="pt")["pixel_values"].to(device=dev, dtype=dt)
        cap = {}
        h = model.encoder.layer[args.layer].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("a", output_tensor(o).detach()))
        with torch.no_grad():
            model(pixel_values=px)
        h.remove()
        regs.append(cap["a"][0, REGISTER_START:REGISTER_END].cpu())
    R = torch.stack(regs)                                   # (n, slots, d)
    slot_norms = torch.linalg.vector_norm(R, dim=-1).mean(dim=0).tolist()
    logging.info("per-slot mean register norm: %s", ["%.1f" % v for v in slot_norms])

    conds = ["clean_top5", "swap_slot", "rescaled_slot", "zero_slot", "random_span"]
    cols = {k: [] for k in conds}
    diag = {"treatment_norm": [], "swap_delta_norm": [], "amplification": [], "slot": []}

    for idx, rec in enumerate(records):
        px = proc(images=rec["image"], return_tensors="pt")["pixel_values"].to(device=dev, dtype=dt)
        cap = {}
        h = model.encoder.layer[args.layer].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("a", output_tensor(o).detach()))
        with torch.no_grad():
            model(pixel_values=px)
        h.remove()
        act = cap["a"]; reg = act[0, REGISTER_START:REGISTER_END]

        with torch.no_grad():
            _x, f = encode_decode(sae, reg)
            ds = selected_contribution(sae, f, coordinates_from_positive_support(f))
            per_slot = torch.linalg.vector_norm(ds, dim=-1)
            slot = int(torch.argmax(per_slot))
            t_norm = float(per_slot[slot])

            rng, _ = deterministic_rng(SEED, "primary", rec["stream_position"], "random_span", 0)
            cand, _i, _c = random_decoder_span(sae, f, rng)
            rs, _a, _b, _e = match_energy(cand, ds)

            zero = torch.zeros_like(ds)
            zero[slot] = reg[slot]                       # subtracting it zeroes the slot

            swaps, rescaled, dnorms = [], [], []
            for draw in range(args.draws):
                g, _ = deterministic_rng(SEED, "primary", rec["stream_position"], "swap", draw)
                j = int(g.integers(0, len(R)))
                while j == idx:
                    j = int(g.integers(0, len(R)))
                other = R[j, slot].to(dev)
                d = torch.zeros_like(ds)
                d[slot] = reg[slot] - other              # wholesale replacement, unscaled
                swaps.append(d)
                dnorms.append(float(torch.linalg.vector_norm(d[slot])))
                rescaled.append(match_energy(d, ds)[0])  # same direction, treatment energy

            deltas = [ds, rs, zero] + swaps + rescaled
            stacked = act.expand(len(deltas), -1, -1).clone()
            for pos, d in enumerate(deltas):
                stacked[pos, REGISTER_START:REGISTER_END] -= d
            out = tail_from_layer(model, torch.cat([act, stacked], dim=0), args.layer)
            clean = out[:1, PATCH_START: PATCH_START + N_PATCHES]
            oth = out[1:, PATCH_START: PATCH_START + N_PATCHES]
            cos = torch.nn.functional.cosine_similarity(clean.expand_as(oth), oth, dim=-1).mean(dim=-1)
            drops = [100.0 * (1.0 - float(v)) for v in cos.cpu().tolist()]

        k = args.draws
        cols["clean_top5"].append(drops[0])
        cols["random_span"].append(drops[1])
        cols["zero_slot"].append(drops[2])
        cols["swap_slot"].append(float(np.mean(drops[3:3 + k])))
        cols["rescaled_slot"].append(float(np.mean(drops[3 + k:3 + 2 * k])))
        md = float(np.mean(dnorms))
        diag["treatment_norm"].append(t_norm)
        diag["swap_delta_norm"].append(md)
        diag["amplification"].append(t_norm / max(md, 1e-9))
        diag["slot"].append(slot)
        if (idx + 1) % 64 == 0:
            logging.info("processed %d/%d (%.0fs)", idx + 1, len(records), time.time() - start)

    a = np.array(cols["clean_top5"])
    summary = {
        "model_name": args.model_name, "layer": args.layer, "n_images": len(records),
        "draws": args.draws,
        "intervened_slot": int(np.bincount(diag["slot"]).argmax()),
        "intervened_slot_is_unanimous": bool(len(set(diag["slot"])) == 1),
        "per_slot_mean_register_norm": slot_norms,
        "treatment_norm_mean": float(np.mean(diag["treatment_norm"])),
        "swap_delta_norm_mean": float(np.mean(diag["swap_delta_norm"])),
        "rescaling_amplification_mean": float(np.mean(diag["amplification"])),
        "rescaling_amplification_range": [float(np.min(diag["amplification"])),
                                          float(np.max(diag["amplification"]))],
        "condition_means": {k: float(np.mean(v)) for k, v in cols.items()},
        "contrasts_top5_minus_control": {},
    }
    for k in ("swap_slot", "rescaled_slot", "zero_slot", "random_span"):
        summary["contrasts_top5_minus_control"][k] = paired_bootstrap(a, np.array(cols[k]), 10000, SEED + 77)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    import csv
    with open(args.output_dir / "per_image_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(conds + ["treatment_norm", "swap_delta_norm", "slot"])
        for i in range(len(records)):
            w.writerow([cols[c][i] for c in conds] +
                       [diag["treatment_norm"][i], diag["swap_delta_norm"][i], diag["slot"][i]])

    logging.info("")
    logging.info("  intervened slot %d (unanimous=%s)", summary["intervened_slot"],
                 summary["intervened_slot_is_unanimous"])
    logging.info("  treatment norm %.2f | swap delta norm %.2f | amplification %.1fx (range %.1f-%.1f)",
                 summary["treatment_norm_mean"], summary["swap_delta_norm_mean"],
                 summary["rescaling_amplification_mean"], *summary["rescaling_amplification_range"])
    for k, v in summary["condition_means"].items():
        logging.info("  %-16s %8.3f", k, v)
    for k, v in summary["contrasts_top5_minus_control"].items():
        logging.info("  top5 - %-14s %+8.3f  CI[%+.3f,%+.3f]  excl0=%s",
                     k, v["mean_difference"], v["ci95_low"], v["ci95_high"], v["excludes_zero"])
    logging.info("done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
