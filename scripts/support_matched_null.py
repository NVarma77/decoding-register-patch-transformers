"""A support-matched null, and the demeaned patch participation ratio.

Two gaps a referee identified, both answerable with the existing harness.

1. Every control in the paper is matched on perturbation *energy* but is drawn
   from a direction the model has never seen at that position: a random decoder
   span of norm ~111 applied to a token of norm ~111 is close to replacing the
   token with noise.  The null the analysis actually calls for is one drawn from
   the empirical distribution at the same slot.  We build it by displacing the
   register toward another image's register at the same slot, rescaled to the
   selected contribution's energy, so the perturbation has the size of the
   treatment and the direction of something the model does encounter.

2. Table 3 reports the corrected within-slot participation ratio for registers
   but a pooled one for patches, while the language-model rows are checked for
   pooling.  Pooling 256 patch positions with different means inflates their
   ratio, so the comparison is not measured the same way on both sides.  We
   report the patch ratio with each position's own mean removed.
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
from scripts.layer_participation_sweep import participation_ratio  # noqa: E402
from scripts.subspace_matched_control import paired_bootstrap  # noqa: E402
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from utils.utils import load_model  # noqa: E402


def pr_of(x: np.ndarray) -> float:
    c = x - x.mean(axis=0, keepdims=True)
    return float(participation_ratio(np.linalg.eigvalsh((c.T @ c) / max(len(c) - 1, 1))[::-1]))


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
    logging.info("n=%d layer=%d", len(records), args.layer)

    # Pass 1: cache register and patch activations.
    regs, patches = [], []
    for rec in records:
        px = proc(images=rec["image"], return_tensors="pt")["pixel_values"].to(device=dev, dtype=dt)
        cap = {}
        h = model.encoder.layer[args.layer].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("a", output_tensor(o).detach()))
        with torch.no_grad():
            model(pixel_values=px)
        h.remove()
        a = cap["a"]
        regs.append(a[0, REGISTER_START:REGISTER_END].cpu())
        patches.append(a[0, PATCH_START: PATCH_START + N_PATCHES].cpu().numpy())
    R = torch.stack(regs)                                  # (n, slots, d)
    P = np.stack(patches)                                  # (n, 256, d)

    patch_pooled = pr_of(P.reshape(-1, P.shape[-1]))
    patch_demeaned = pr_of((P - P.mean(axis=0, keepdims=True)).reshape(-1, P.shape[-1]))
    logging.info("patch PR pooled %.2f | position-demeaned %.2f", patch_pooled, patch_demeaned)

    # Pass 2: the support-matched null against the treatment.
    cols = {"clean_top5": [], "support_matched": [], "random_span": []}
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
            rng, _ = deterministic_rng(SEED, "primary", rec["stream_position"], "random_span", 0)
            cand, _i, _c = random_decoder_span(sae, f, rng)
            rs, _a, _b, _e = match_energy(cand, ds)

            # Support-matched: displace toward another image's register at the
            # same slot, then rescale to the treatment's energy. Direction is
            # one the model encounters; magnitude matches the treatment.
            deltas = [ds, rs]
            for draw in range(args.draws):
                g, _ = deterministic_rng(SEED, "primary", rec["stream_position"], "support", draw)
                j = int(g.integers(0, len(R)))
                while j == idx:
                    j = int(g.integers(0, len(R)))
                cand2 = (reg - R[j].to(dev))
                sm, _a, _b, _e = match_energy(cand2, ds)
                deltas.append(sm)

            stacked = act.expand(len(deltas), -1, -1).clone()
            for pos, d in enumerate(deltas):
                stacked[pos, REGISTER_START:REGISTER_END] -= d
            out = tail_from_layer(model, torch.cat([act, stacked], dim=0), args.layer)
            clean = out[:1, PATCH_START: PATCH_START + N_PATCHES]
            other = out[1:, PATCH_START: PATCH_START + N_PATCHES]
            cos = torch.nn.functional.cosine_similarity(clean.expand_as(other), other, dim=-1).mean(dim=-1)
            drops = [100.0 * (1.0 - float(v)) for v in cos.cpu().tolist()]
        cols["clean_top5"].append(drops[0])
        cols["random_span"].append(drops[1])
        cols["support_matched"].append(float(np.mean(drops[2:])))
        if (idx + 1) % 64 == 0:
            logging.info("processed %d/%d (%.0fs)", idx + 1, len(records), time.time() - start)

    a = np.array(cols["clean_top5"])
    summary = {"model_name": args.model_name, "layer": args.layer, "n_images": len(records),
               "draws": args.draws,
               "patch_pr_pooled": patch_pooled, "patch_pr_position_demeaned": patch_demeaned,
               "condition_means": {k: float(np.mean(v)) for k, v in cols.items()},
               "contrasts_top5_minus_control": {}}
    for k in ("support_matched", "random_span"):
        b = np.array(cols[k])
        summary["contrasts_top5_minus_control"][k] = paired_bootstrap(a, b, 10000, SEED + 77)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    logging.info("")
    for k, v in summary["condition_means"].items():
        logging.info("  %-16s %8.3f", k, v)
    for k, v in summary["contrasts_top5_minus_control"].items():
        logging.info("  top5 - %-14s %+8.3f  CI[%+.3f,%+.3f]  excl0=%s",
                     k, v["mean_difference"], v["ci95_low"], v["ci95_high"], v["excludes_zero"])
    logging.info("done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
