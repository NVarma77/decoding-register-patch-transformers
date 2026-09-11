"""Is the register cloud's low participation ratio an artifact of pooling slots?

A referee observed that the register PCA pools all four register positions, and
that its leading component largely separates one fixed slot from the other
three.  If that is what the leading component encodes, then a pooled
participation ratio near one measures a between-slot mean offset rather than a
low-dimensional cloud, and the intervened slot's own distribution -- the one
that matters, since the intervention always targets the same slot -- may not be
degenerate at all.

This script decides it by computing three quantities per layer:

  pooled            PR over all register vectors, which is what the paper reports
  per_slot          PR within each register slot separately, across images
  demeaned          PR over all register vectors after subtracting each slot's
                    own mean, which removes the between-slot offset and keeps
                    only within-slot variation

If per_slot and demeaned are also near one, the pooled number is not an
artifact and the geometric account stands as written.  If they are much larger,
the paper's central quantity is measuring slot identity and the on-manifold
framing has to be rebuilt on the intervened slot's own distribution.
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
    make_cfg,
    output_tensor,
)
from scripts.layer_participation_sweep import participation_ratio  # noqa: E402
from utils.utils import load_model  # noqa: E402


def pr_of(x: np.ndarray) -> float:
    """Participation ratio of the centered covariance of ``x`` (rows are vectors)."""
    c = x - x.mean(axis=0, keepdims=True)
    cov = (c.T @ c) / max(len(c) - 1, 1)
    return float(participation_ratio(np.linalg.eigvalsh(cov)[::-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-ids", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--cohort-cache", type=Path, required=True)
    ap.add_argument("--allow-missing", type=int, default=2)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--n-images", type=int, default=511)
    ap.add_argument("--batch-size", type=int, default=16)
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
    n_layers = len(model.encoder.layer)
    n_slots = REGISTER_END - REGISTER_START
    logging.info("model=%s n=%d layers=%d slots=%d", args.model_name,
                 len(records), n_layers, n_slots)

    # (layer, image, slot, d)
    buf: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    patch_buf: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    for s in range(0, len(records), args.batch_size):
        chunk = records[s: s + args.batch_size]
        px = processor(images=[r["image"] for r in chunk],
                       return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
        cap: dict[int, torch.Tensor] = {}
        handles = [model.encoder.layer[l].register_forward_hook(
            lambda _m, _i, o, l=l: cap.__setitem__(l, output_tensor(o).detach()))
            for l in range(n_layers)]
        with torch.no_grad():
            model(pixel_values=px)
        for h in handles:
            h.remove()
        for l in range(n_layers):
            a = cap[l]
            buf[l].append(a[:, REGISTER_START:REGISTER_END].float().cpu().numpy())
            patch_buf[l].append(
                a[:, PATCH_START: PATCH_START + N_PATCHES].float().cpu().numpy())
        if (s // args.batch_size) % 8 == 0:
            logging.info("processed %d/%d (%.0fs)", s + len(chunk), len(records),
                         time.time() - start)

    per_layer = []
    for l in range(n_layers):
        reg = np.concatenate(buf[l])                      # (n_img, slots, d)
        pooled = pr_of(reg.reshape(-1, reg.shape[-1]))
        per_slot = [pr_of(reg[:, i]) for i in range(n_slots)]
        demeaned = pr_of((reg - reg.mean(axis=0, keepdims=True)).reshape(-1, reg.shape[-1]))
        patch = np.concatenate(patch_buf[l])
        entry = {
            "layer": l,
            "pooled_pr": pooled,
            "per_slot_pr": per_slot,
            "demeaned_pr": demeaned,
            "patch_pr": pr_of(patch.reshape(-1, patch.shape[-1])[:60000]),
        }
        per_layer.append(entry)
        logging.info("L%-2d pooled %8.3f | per-slot %s | demeaned %8.3f",
                     l, pooled, " ".join("%7.3f" % v for v in per_slot), demeaned)

    summary = {"model_name": args.model_name, "n_images": len(records),
               "n_slots": n_slots, "per_layer": per_layer}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    logging.info("done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
