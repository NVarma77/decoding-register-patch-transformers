"""Does register attention mass order the intervention conditions?

Registers act as attention sinks, so a rival account of the layer-8 result says
the ordering is set by how much attention mass the perturbed register absorbs,
not by where the perturbation points relative to the occupied region.  The
resulting-norm form of that account is already falsified by the packaged data:
the condition with the smallest norm change (random, +41%) is the most
disruptive, and the one with the largest (addition, +100%) is the least.  But
attention mass depends on key and query projections rather than raw norm, so
rescaling a register can preserve its attention pattern while a random rotation
scrambles it.  This script measures attention mass directly.

For each image we capture the layer-8 activation, build the four conditions,
and re-execute block 9 with output_attentions=True.  We report the attention
probability that patch queries place on the perturbed register key, averaged
over heads and patch queries.

Reading:
  * If attention mass orders the conditions the way cost does, the sink account
    is live and the geometric account is not separated from it.
  * If it does not -- in particular if addition moves attention mass a long way
    at almost no downstream cost -- then attention mass is not what the cost is
    tracking, and the same reasoning that killed the norm form kills this one.
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
    MODEL_NAME,
    N_PATCHES,
    PATCH_START,
    REGISTER_END,
    REGISTER_START,
    SEED,
    coordinates_from_positive_support,
    deterministic_rng,
    encode_decode,
    make_cfg,
    match_energy,
    output_tensor,
    random_decoder_span,
    selected_contribution,
)
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from utils.utils import load_model  # noqa: E402

CONDITIONS = ("clean", "removal", "addition", "random")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-ids", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--cohort-cache", type=Path, required=True)
    ap.add_argument("--allow-missing", type=int, default=2)
    ap.add_argument("--sae", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n-images", type=int, default=128)
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
    device = torch.device(args.device)
    dtype = torch.float32

    from scripts.cohort_local import load_cohort
    records = load_cohort(args.image_ids, args.workdir, args.cohort_cache,
                          args.allow_missing)[: args.n_images]

    model, _tok, processor = load_model(
        args.model_name, make_cfg(str(device), dtype, args.model_name),
        dtype=dtype, device=str(device))
    model.eval()
    sae = AutoEncoderTopK.from_pretrained(str(args.sae / "ae.pt"), device=str(device))
    logging.info("n=%d layer=%d", len(records), args.layer)

    mass_perturbed = {c: [] for c in CONDITIONS}   # patch queries -> perturbed register key
    mass_all_reg = {c: [] for c in CONDITIONS}     # patch queries -> all four register keys
    norms = {c: [] for c in CONDITIONS}

    for record in records:
        px = processor(images=record["image"], return_tensors="pt")["pixel_values"].to(
            device=device, dtype=dtype)
        cap = {}
        h = model.encoder.layer[args.layer].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("a", output_tensor(o).detach()))
        with torch.no_grad():
            model(pixel_values=px)
        h.remove()
        act = cap["a"]
        registers = act[0, REGISTER_START:REGISTER_END]

        with torch.no_grad():
            _xhat, features = encode_decode(sae, registers)
            selected = coordinates_from_positive_support(features)
            ds = selected_contribution(sae, features, selected)
            rng, _ = deterministic_rng(SEED, "primary", record["stream_position"],
                                       "random_span", 0)
            cand, _ids, _coef = random_decoder_span(sae, features, rng)
            rnd, _t, _u, _a = match_energy(cand, ds)

            # The row carrying the selected latents is the one under test.
            row = int(torch.linalg.vector_norm(ds, dim=-1).argmax())
            deltas = {"clean": torch.zeros_like(ds), "removal": ds,
                      "addition": -ds, "random": rnd}

            stacked = act.expand(len(CONDITIONS), -1, -1).clone()
            for i, c in enumerate(CONDITIONS):
                stacked[i, REGISTER_START:REGISTER_END] -= deltas[c]

            # Re-execute the block immediately after the hook, asking for its
            # attention probabilities. This is the first block whose attention
            # can see the perturbed key.
            out = model.encoder.layer[args.layer + 1](stacked, None, True)
            attn = out[1]  # (cond, heads, query, key)

            patch_q = attn[:, :, PATCH_START: PATCH_START + N_PATCHES, :]
            to_perturbed = patch_q[:, :, :, REGISTER_START + row]
            to_all_reg = patch_q[:, :, :, REGISTER_START:REGISTER_END].sum(dim=-1)

            for i, c in enumerate(CONDITIONS):
                mass_perturbed[c].append(float(to_perturbed[i].mean()))
                mass_all_reg[c].append(float(to_all_reg[i].mean()))
                norms[c].append(float(torch.linalg.vector_norm(
                    stacked[i, REGISTER_START + row])))

        if len(mass_perturbed["clean"]) % 32 == 0:
            logging.info("processed %d/%d (%.0fs)", len(mass_perturbed["clean"]),
                         len(records), time.time() - start)

    # Downstream costs are measured elsewhere; quoted here only to compare orderings.
    cost = {"clean": 0.0, "removal": 7.364, "addition": 0.227, "random": 14.155}
    summary = {"model_name": args.model_name, "layer": args.layer,
               "n_images": len(records), "attention_block": args.layer + 1,
               "downstream_cost_reference": cost, "conditions": {}}
    for c in CONDITIONS:
        summary["conditions"][c] = {
            "attn_mass_to_perturbed_register": float(np.mean(mass_perturbed[c])),
            "attn_mass_to_all_registers": float(np.mean(mass_all_reg[c])),
            "perturbed_register_norm": float(np.mean(norms[c])),
        }
    base = summary["conditions"]["clean"]["attn_mass_to_perturbed_register"]
    for c in CONDITIONS:
        m = summary["conditions"][c]["attn_mass_to_perturbed_register"]
        summary["conditions"][c]["attn_mass_change_pct"] = float(100.0 * (m - base) / base)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    logging.info("")
    logging.info("%-9s %14s %12s %14s %10s", "condition", "attn->perturbed",
                 "change", "attn->all reg", "cost")
    for c in CONDITIONS:
        e = summary["conditions"][c]
        logging.info("%-9s %14.6f %11.1f%% %14.6f %10.3f", c,
                     e["attn_mass_to_perturbed_register"], e["attn_mass_change_pct"],
                     e["attn_mass_to_all_registers"], cost[c])
    order_cost = [c for c in CONDITIONS if c != "clean"]
    order_cost.sort(key=lambda c: cost[c])
    order_attn = [c for c in CONDITIONS if c != "clean"]
    order_attn.sort(key=lambda c: abs(summary["conditions"][c]["attn_mass_change_pct"]))
    logging.info("")
    logging.info("cost order (low to high):            %s", " < ".join(order_cost))
    logging.info("|attn mass change| (low to high):    %s", " < ".join(order_attn))
    logging.info("orderings agree: %s", order_cost == order_attn)
    logging.info("done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
