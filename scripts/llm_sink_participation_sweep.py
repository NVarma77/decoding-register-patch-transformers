"""Experiment D -- is the register geometry a special case of a general pattern?

Experiments A-C establish, for DINOv2 registers, that the activation cloud of a
special token population collapses to near rank-one across a band of layers, and
that where it does, ablating its dominant direction measures geometry rather
than feature content.

The obvious objection is scope: who intervenes on four register tokens?  The
failure mode only matters if it describes populations people actually probe.

Attention-sink tokens in language models are exactly such a population.  The
first position absorbs disproportionate attention mass (Xiao et al.'s
StreamingLLM) and carries outsized-norm activations (Sun et al.'s massive
activations), and it is routinely the subject of interpretability claims.  If
its activation cloud is also near rank-one, then the diagnostic proposed here is
architecture-general rather than a DINOv2 curiosity.

This runs the *identical* measurement as Experiment A -- participation ratio of a
token population's activation cloud, per layer -- with the sink position playing
the role of the register and ordinary later positions playing the role of
patches.  No SAE and no intervention: one forward pass per sequence, all layers
hooked at once, second moments accumulated as d x d Gram matrices in float64 on
device.

Prediction, fixed before running: if the register result generalises, the sink
cloud should show a markedly lower participation ratio than ordinary token
positions at the same layers.  If sink and ordinary clouds have comparable
effective dimensionality, the register finding is specific to ViT registers and
the paper must say so.
"""

from __future__ import annotations

import argparse
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

from scripts.layer_participation_sweep import (  # noqa: E402
    SUBSPACE_DIMS,
    MomentAccumulator,
    participation_ratio,
)

SEED = 20260822


def load_texts(n_sequences: int, min_chars: int) -> list[str]:
    """Wikitext-103 validation paragraphs, long enough to fill the context."""
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts: list[str] = []
    for row in data:
        text = row["text"].strip()
        if len(text) >= min_chars and not text.startswith("="):
            texts.append(text)
        if len(texts) >= n_sequences:
            break
    if len(texts) < n_sequences:
        raise RuntimeError(f"Only {len(texts)} usable paragraphs; wanted {n_sequences}")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-sequences", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ordinary-start",
        type=int,
        default=8,
        help="First position counted as an ordinary token; skips the early positions "
        "adjacent to the sink so the two clouds are cleanly separated.",
    )
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, output_hidden_states=True
    ).to(device)
    model.eval()

    texts = load_texts(args.n_sequences, args.min_chars)
    logging.info("model=%s sequences=%d seq_len=%d", args.model_name, len(texts), args.seq_len)

    n_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    logging.info("layers=%d hidden=%d", n_layers, dim)

    # hidden_states has n_layers+1 entries (embeddings + each block output); we
    # take block outputs so the indexing matches Experiment A's "layer i output".
    sink_acc = [MomentAccumulator(dim, device) for _ in range(n_layers)]
    ordinary_acc = [MomentAccumulator(dim, device) for _ in range(n_layers)]
    sink_norms: list[float] = []
    ordinary_norms: list[float] = []

    for start in range(0, len(texts), args.batch_size):
        chunk = texts[start : start + args.batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        # Only use sequences that genuinely fill the context, so padding never
        # enters either cloud.
        keep = attention_mask.sum(dim=1) == args.seq_len
        if not bool(keep.any()):
            continue
        input_ids, attention_mask = input_ids[keep], attention_mask[keep]

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        for layer in range(n_layers):
            hidden = out.hidden_states[layer + 1]
            sink = hidden[:, 0:1]
            ordinary = hidden[:, args.ordinary_start :]
            sink_acc[layer].update(sink)
            ordinary_acc[layer].update(ordinary)
            if layer == n_layers // 2:
                sink_norms.extend(sink.reshape(-1, dim).norm(dim=-1).cpu().tolist())
                ordinary_norms.extend(
                    ordinary.reshape(-1, dim).norm(dim=-1).cpu().tolist()
                )
        if (start // args.batch_size) % 8 == 0:
            logging.info("processed %d/%d sequences", start + len(chunk), len(texts))

    per_layer: list[dict[str, Any]] = []
    for layer in range(n_layers):
        eig_sink = np.linalg.eigvalsh(sink_acc[layer].covariance())[::-1]
        eig_ord = np.linalg.eigvalsh(ordinary_acc[layer].covariance())[::-1]
        total = float(eig_sink[eig_sink > 0].sum())
        entry = {
            "layer": layer,
            "sink_participation_ratio": participation_ratio(eig_sink),
            "ordinary_participation_ratio": participation_ratio(eig_ord),
            "sink_top1_variance_fraction": float(eig_sink[0] / total) if total > 0 else 0.0,
            "n_sink_vectors": sink_acc[layer].count,
            "n_ordinary_vectors": ordinary_acc[layer].count,
            "sink_cumulative_variance_at_dim": {
                str(d): float(eig_sink[:d][eig_sink[:d] > 0].sum() / total) if total > 0 else 0.0
                for d in SUBSPACE_DIMS
            },
        }
        per_layer.append(entry)
        logging.info(
            "layer %2d  sink PR=%8.3f  ordinary PR=%8.3f  top1=%.4f",
            layer,
            entry["sink_participation_ratio"],
            entry["ordinary_participation_ratio"],
            entry["sink_top1_variance_fraction"],
        )

    summary = {
        "model_name": args.model_name,
        "n_layers": n_layers,
        "hidden_size": dim,
        "seq_len": args.seq_len,
        "ordinary_start": args.ordinary_start,
        "mean_sink_norm_midlayer": float(np.mean(sink_norms)) if sink_norms else None,
        "mean_ordinary_norm_midlayer": float(np.mean(ordinary_norms)) if ordinary_norms else None,
        "per_layer": per_layer,
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if sink_norms:
        logging.info(
            "mid-layer norm: sink %.1f vs ordinary %.1f (ratio %.1fx)",
            summary["mean_sink_norm_midlayer"],
            summary["mean_ordinary_norm_midlayer"],
            summary["mean_sink_norm_midlayer"] / max(summary["mean_ordinary_norm_midlayer"], 1e-9),
        )
    logging.info("done in %.1fs", time.time() - start_time)


if __name__ == "__main__":
    main()
