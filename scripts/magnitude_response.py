"""Does the downstream endpoint saturate with perturbation magnitude?

A referee asked why Section 7's off-manifold cost at layer 8 (14.006) nearly
equals Section 6's random-decoder-span cost (14.155) when the two protocols
match energy to *different* targets: Section 6 to the SAE's selected
contribution, Section 7 to the register's own centered leading component.  If
those two targets differ in norm, near-equal cost is evidence that the
off-manifold condition is saturated in magnitude, and a saturated numerator
would compromise the ratio A = log2(off/on) used across layers.

This script answers it directly at layer 8 of DINOv2-Small:

  1. Reports the per-row norms of the two targets and their ratio.
  2. Sweeps a scale factor over both an aligned-V1 (on-manifold) removal and a
     random off-manifold direction, holding everything else fixed, and reports
     the cost curve for each.

If the random curve is flat over the range spanned by the two protocols, the
endpoint saturates and the paper must say so.  If it rises, the two protocols
simply sit at different points on a monotone curve and the near-coincidence is
a numerical accident that the paper should still explain.
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
    output_tensor,
    selected_contribution,
)
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from utils.utils import load_model  # noqa: E402


def tail_from_layer(model, hidden, layer):
    for index in range(layer + 1, len(model.encoder.layer)):
        hidden = model.encoder.layer[index](hidden, None, False)[0]
    return model.layernorm(hidden)


def unit(v):
    return v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-ids", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--cohort-cache", type=Path, required=True)
    ap.add_argument("--allow-missing", type=int, default=2)
    ap.add_argument("--sae", type=Path, required=True)
    ap.add_argument("--basis", type=Path, required=True, help="register_basis_intervention_layer.npz")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--random-draws", type=int, default=4)
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
    records = load_cohort(args.image_ids, args.workdir, args.cohort_cache, args.allow_missing)[: args.n_images]
    logging.info("n=%d layer=%d scales=%s", len(records), args.layer, args.scales)

    npz = np.load(args.basis)
    V = torch.as_tensor(npz["eigenvectors"][:, :1].copy(), dtype=dtype, device=device)
    mu = torch.as_tensor(npz["mean"].copy(), dtype=dtype, device=device)

    model, _tok, processor = load_model(
        args.model_name, make_cfg(str(device), dtype, args.model_name), dtype=dtype, device=str(device))
    model.eval()
    sae = AutoEncoderTopK.from_pretrained(str(args.sae / "ae.pt"), device=str(device))

    names = ["aligned_s%g" % s for s in args.scales] + \
            ["random_s%g" % s for s in args.scales]
    rows = []
    norm_sel, norm_on = [], []

    for record in records:
        px = processor(images=record["image"], return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
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
            delta_selected = selected_contribution(sae, features, selected)

            centered = registers - mu
            delta_on = (centered @ V) @ V.T

            n_sel = torch.linalg.vector_norm(delta_selected, dim=-1)
            n_on = torch.linalg.vector_norm(delta_on, dim=-1)
            norm_sel.append(n_sel.cpu().numpy())
            norm_on.append(n_on.cpu().numpy())

            # Both families are scaled relative to the SELECTED contribution's
            # norm, so scale 1.0 reproduces the Section 6 energy exactly.
            aligned_dir = unit((delta_selected @ V) @ V.T)
            deltas = [aligned_dir * (s * n_sel).unsqueeze(-1) for s in args.scales]

            # Each random draw is evaluated separately and averaged over costs,
            # not over vectors: averaging unit vectors would shrink the norm.
            rand_blocks = []
            for s in args.scales:
                for draw in range(args.random_draws):
                    rng, _ = deterministic_rng(SEED, "magsweep", record["stream_position"], "rand", draw)
                    cand = torch.as_tensor(rng.normal(size=registers.shape).astype(np.float32),
                                           dtype=dtype, device=device)
                    rand_blocks.append(unit(cand) * (s * n_sel).unsqueeze(-1))
            deltas = deltas + rand_blocks

            stacked = act.expand(len(deltas), -1, -1).clone()
            for pos, d in enumerate(deltas):
                stacked[pos, REGISTER_START:REGISTER_END] -= d
            out = tail_from_layer(model, torch.cat([act, stacked], dim=0), args.layer)
            clean = out[:1, PATCH_START: PATCH_START + N_PATCHES]
            other = out[1:, PATCH_START: PATCH_START + N_PATCHES]
            cos = torch.nn.functional.cosine_similarity(clean.expand_as(other), other, dim=-1).mean(dim=-1)
            drops = [100.0 * (1.0 - float(v)) for v in cos.cpu().tolist()]

        row = {"stream_position": record["stream_position"]}
        for i, s in enumerate(args.scales):
            row["aligned_s%g" % s] = drops[i]
        base = len(args.scales)
        for i, s in enumerate(args.scales):
            block = drops[base + i * args.random_draws: base + (i + 1) * args.random_draws]
            row["random_s%g" % s] = float(np.mean(block))
        rows.append(row)
        if len(rows) % 32 == 0:
            logging.info("processed %d/%d (%.0fs)", len(rows), len(records), time.time() - start)

    sel = np.concatenate(norm_sel)
    on = np.concatenate(norm_on)
    summary = {
        "model_name": args.model_name,
        "layer": args.layer,
        "n_images": len(rows),
        "random_draws": args.random_draws,
        "selected_contribution_norm_mean": float(sel.mean()),
        "centered_pc1_component_norm_mean": float(on.mean()),
        "norm_ratio_pc1_over_selected": float(on.mean() / sel.mean()),
        "scales": args.scales,
        "curves": {},
    }
    for family in ("aligned", "random"):
        summary["curves"][family] = [
            {"scale": s, "mean_cost": float(np.mean([r["%s_s%g" % (family, s)] for r in rows]))}
            for s in args.scales
        ]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    logging.info("||delta_selected||=%.3f  ||centered PC1||=%.3f  ratio=%.3f",
                 sel.mean(), on.mean(), on.mean() / sel.mean())
    for family in ("aligned", "random"):
        logging.info("%s:  %s", family,
                     "  ".join("s=%.2f:%.3f" % (e["scale"], e["mean_cost"])
                               for e in summary["curves"][family]))
    logging.info("done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
