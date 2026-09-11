"""Experiment A -- participation ratio of the register and patch clouds, per layer.

The paper measures effective dimensionality only at layer 8 (register PR 1.016
vs patch PR 120.6).  Every geometric claim therefore rests on a single depth.
This sweep measures the same quantity at every block, which costs no SAE and no
intervention: one forward pass per image, with all blocks hooked at once.

Design notes
------------
GPU is the compute engine.  All twelve block outputs are captured in a *single*
forward pass per batch, so depth is free -- the sweep costs the same as
measuring one layer.  Second moments are accumulated as 384x384 (or 768x768)
Gram matrices in float64 on device, so the full patch cloud (N_images x 256
vectors per layer) never has to be materialised in host memory; only the
accumulators are kept.  Eigendecomposition runs once at the end.

Parallelism is bounded by the box, not by ambition: 4 CPU cores and one L4.
JPEG decode and preprocessing are the only CPU-bound stage and run on a
3-worker thread pool feeding a prefetch queue; the GPU consumes whole batches.
Spawning multiple GPU processes would contend for one device and starve on four
cores, so it is deliberately not done.

Eigenvectors at the intervention layer are written out for reuse by
Experiment B (the subspace-matched control), which needs the register cloud's
principal basis.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
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
    make_cfg,
    output_tensor,
)
from utils.utils import load_model  # noqa: E402

SUBSPACE_DIMS = [1, 2, 3, 4, 8, 16, 32, 64]


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """(sum l)^2 / sum l^2 -- identical to register_subspace_geometry.py."""
    positive = eigenvalues[eigenvalues > 0]
    if positive.size == 0:
        return 0.0
    return float(positive.sum() ** 2 / (positive**2).sum())


class MomentAccumulator:
    """Streaming mean and second moment for one activation cloud, on device."""

    def __init__(self, dim: int, device: torch.device) -> None:
        self.count = 0
        self.total = torch.zeros(dim, dtype=torch.float64, device=device)
        self.gram = torch.zeros((dim, dim), dtype=torch.float64, device=device)

    def update(self, batch: torch.Tensor) -> None:
        x = batch.reshape(-1, batch.shape[-1]).to(torch.float64)
        self.count += x.shape[0]
        self.total += x.sum(dim=0)
        self.gram += x.T @ x

    def covariance(self) -> np.ndarray:
        """Unbiased covariance, matching the (n-1) convention used in the paper."""
        if self.count < 2:
            raise RuntimeError("Need at least two vectors for a covariance")
        mean = self.total / self.count
        centered_gram = self.gram - self.count * torch.outer(mean, mean)
        return (centered_gram / (self.count - 1)).cpu().numpy()


def decode_worker(
    records: list[dict[str, Any]],
    processor: Any,
    batch_size: int,
    out: queue.Queue,
    n_workers: int,
) -> None:
    """Decode and preprocess batches on CPU threads, feeding the GPU."""
    lock = threading.Lock()
    cursor = {"i": 0}

    def run() -> None:
        while True:
            with lock:
                start = cursor["i"]
                if start >= len(records):
                    return
                cursor["i"] = start + batch_size
            chunk = records[start : start + batch_size]
            pixels = processor(images=[r["image"] for r in chunk], return_tensors="pt")["pixel_values"]
            out.put((start, pixels))

    threads = [threading.Thread(target=run, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.put(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cohort-cache", type=Path, default=None)
    parser.add_argument("--allow-missing", type=int, default=0)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=3)
    parser.add_argument("--intervention-layer", type=int, default=8)
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
    logging.info("device=%s model=%s batch=%d", device, args.model_name, args.batch_size)

    from scripts.cohort_local import load_cohort

    records = load_cohort(args.image_ids, args.workdir, args.cohort_cache, args.allow_missing)
    logging.info("cohort=%d images", len(records))

    model, _tok, processor = load_model(
        args.model_name, make_cfg(str(device), dtype, args.model_name), dtype=dtype, device=str(device)
    )
    model.eval()
    blocks = model.encoder.layer
    n_layers = len(blocks)
    dim = model.config.hidden_size
    logging.info("layers=%d hidden=%d", n_layers, dim)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(_m, _i, output):
            captured[index] = output_tensor(output).detach()
            return output

        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]

    reg_acc = [MomentAccumulator(dim, device) for _ in range(n_layers)]
    patch_acc = [MomentAccumulator(dim, device) for _ in range(n_layers)]

    pending: queue.Queue = queue.Queue(maxsize=4)
    producer = threading.Thread(
        target=decode_worker,
        args=(records, processor, args.batch_size, pending, args.decode_workers),
        daemon=True,
    )
    producer.start()

    done = 0
    while True:
        item = pending.get()
        if item is None:
            break
        _offset, pixels = item
        pixels = pixels.to(device=device, dtype=dtype, non_blocking=True)
        captured.clear()
        with torch.no_grad():
            model(pixel_values=pixels)
        for layer in range(n_layers):
            act = captured[layer]
            if act.shape[1] != SEQUENCE_LENGTH:
                raise RuntimeError(f"Unexpected sequence length {act.shape[1]} at layer {layer}")
            reg_acc[layer].update(act[:, REGISTER_START:REGISTER_END])
            patch_acc[layer].update(act[:, PATCH_START : PATCH_START + N_PATCHES])
        done += pixels.shape[0]
        if done % (args.batch_size * 4) == 0:
            logging.info("processed %d/%d images", done, len(records))
    producer.join()
    for handle in handles:
        handle.remove()
    logging.info("forward passes complete in %.1fs", time.time() - start_time)

    per_layer: list[dict[str, Any]] = []
    all_bases: dict[str, np.ndarray] = {}
    for layer in range(n_layers):
        cov_reg = reg_acc[layer].covariance()
        cov_patch = patch_acc[layer].covariance()
        eig_reg, vec_reg = np.linalg.eigh(cov_reg)
        order = np.argsort(eig_reg)[::-1]
        eig_reg, vec_reg = eig_reg[order], vec_reg[:, order]
        eig_patch = np.linalg.eigvalsh(cov_patch)[::-1]

        total = float(eig_reg[eig_reg > 0].sum())
        entry = {
            "layer": layer,
            "register_participation_ratio": participation_ratio(eig_reg),
            "patch_participation_ratio": participation_ratio(eig_patch),
            "register_top1_variance_fraction": float(eig_reg[0] / total) if total > 0 else 0.0,
            "n_register_vectors": reg_acc[layer].count,
            "n_patch_vectors": patch_acc[layer].count,
            "register_cumulative_variance_at_dim": {
                str(d): float(eig_reg[:d][eig_reg[:d] > 0].sum() / total) if total > 0 else 0.0
                for d in SUBSPACE_DIMS
            },
        }
        per_layer.append(entry)
        logging.info(
            "layer %2d  register PR=%8.3f  patch PR=%8.3f  top1=%.4f",
            layer,
            entry["register_participation_ratio"],
            entry["patch_participation_ratio"],
            entry["register_top1_variance_fraction"],
        )

        mean_vec = (reg_acc[layer].total / reg_acc[layer].count).cpu().numpy()
        all_bases[f"eigenvectors_{layer}"] = vec_reg
        all_bases[f"eigenvalues_{layer}"] = eig_reg
        all_bases[f"mean_{layer}"] = mean_vec
        if layer == args.intervention_layer:
            np.savez(
                args.output_dir / "register_basis_intervention_layer.npz",
                eigenvalues=eig_reg,
                eigenvectors=vec_reg,
                mean=mean_vec,
                layer=layer,
            )
            logging.info("saved register principal basis for layer %d", layer)

    np.savez(args.output_dir / "register_bases_all_layers.npz", n_layers=n_layers, **all_bases)
    logging.info("saved register principal bases for all %d layers", n_layers)

    summary = {
        "model_name": args.model_name,
        "n_images": len(records),
        "n_layers": n_layers,
        "activation_dim": dim,
        "intervention_layer": args.intervention_layer,
        "image_ids_source": str(args.image_ids),
        "elapsed_seconds": time.time() - start_time,
        "per_layer": per_layer,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("wrote %s (%.1fs total)", args.output_dir / "summary.json", time.time() - start_time)


if __name__ == "__main__":
    main()
