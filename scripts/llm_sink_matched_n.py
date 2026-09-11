"""Is the language-model sink/ordinary dimensionality gap a sample-size artifact?

The sink cloud has one vector per sequence (~490) in a residual stream of 768 to
896 dimensions, so its sample covariance is rank-deficient, while the ordinary
cloud has ~59,000 vectors and is well estimated.  Participation ratio is
downward biased when n is small relative to d, and the bias runs in the
direction that favours a low sink PR, so the comparison as originally run is
not sample-size matched.

This script recomputes the ordinary-token participation ratio at exactly the
sink's sample size, by taking a single ordinary token per sequence, and reports
both estimates side by side.  If the matched-n ordinary PR stays far above the
sink PR, the gap is not an artifact.  If it collapses toward the sink, it is.
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

from scripts.layer_participation_sweep import participation_ratio  # noqa: E402
from scripts.llm_sink_participation_sweep import load_texts  # noqa: E402

SEED = 20260822


def pr_from(vectors: np.ndarray) -> float:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    return participation_ratio(np.linalg.eigvalsh(cov)[::-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-sequences", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--ordinary-start", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=20, help="matched-n draws")
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

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, output_hidden_states=True).to(device)
    model.eval()

    texts = load_texts(args.n_sequences, args.min_chars)
    dim = model.config.hidden_size
    logging.info("model=%s layer=%d d=%d", args.model_name, args.layer, dim)

    sink_rows, ordinary_rows = [], []
    for s in range(0, len(texts), args.batch_size):
        enc = tok(texts[s: s + args.batch_size], return_tensors="pt", truncation=True,
                  max_length=args.seq_len, padding="max_length")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        keep = mask.sum(dim=1) == args.seq_len
        if not bool(keep.any()):
            continue
        ids, mask = ids[keep], mask[keep]
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        hidden = out.hidden_states[args.layer + 1]
        sink_rows.append(hidden[:, 0].float().cpu().numpy())
        ordinary_rows.append(hidden[:, args.ordinary_start:].float().cpu().numpy())

    sink = np.concatenate(sink_rows)                                  # (n_seq, d)
    ordinary_full = np.concatenate(ordinary_rows)                     # (n_seq*T, d)
    n_sink = len(sink)

    pr_sink = pr_from(sink)
    pr_ord_full = pr_from(ordinary_full.reshape(-1, dim))

    # The ordinary cloud pools positions 8..T-1, whose means differ with
    # position. Pooling subpopulations with different means inflates the
    # participation ratio, exactly the artifact that invalidates a pooled
    # register statistic in the vision case, so we also report the ratio after
    # subtracting each position's own mean. The sink is a single position and
    # cannot suffer from this.
    ord_demeaned = ordinary_full - ordinary_full.mean(axis=0, keepdims=True)
    pr_ord_demeaned = pr_from(ord_demeaned.reshape(-1, dim))

    # Matched-n must match the sink's *structure*, not just its count: the sink
    # contributes exactly one vector per sequence, and ordinary tokens within a
    # sequence are correlated. Drawing uniformly from the pooled tokens would
    # take many tokens from the same sequence and is not the right control, so
    # we draw one ordinary token per sequence instead.
    rng = np.random.default_rng(SEED)
    flat = ordinary_full.reshape(-1, dim)
    n_seq, n_tok = ordinary_full.shape[0], ordinary_full.shape[1]
    matched = []
    for _ in range(args.repeats):
        pick = rng.integers(0, n_tok, size=n_seq)
        matched.append(pr_from(ordinary_full[np.arange(n_seq), pick]))
    pooled = [pr_from(flat[rng.choice(len(flat), size=n_sink, replace=False)])
              for _ in range(args.repeats)]

    summary = {
        "model_name": args.model_name,
        "layer": args.layer,
        "hidden_size": dim,
        "n_sink_vectors": int(n_sink),
        "n_ordinary_vectors_full": int(len(flat)),
        "sink_participation_ratio": float(pr_sink),
        "ordinary_participation_ratio_full_n": float(pr_ord_full),
        "ordinary_participation_ratio_position_demeaned": float(pr_ord_demeaned),
        "gap_position_demeaned": float(pr_ord_demeaned / pr_sink),
        "ordinary_participation_ratio_matched_n_mean": float(np.mean(matched)),
        "ordinary_participation_ratio_matched_n_min": float(np.min(matched)),
        "ordinary_participation_ratio_matched_n_max": float(np.max(matched)),
        "matched_n_repeats": args.repeats,
        "ordinary_pr_pooled_draw_mean": float(np.mean(pooled)),
        "ordinary_pr_pooled_draw_min": float(np.min(pooled)),
        "ordinary_pr_pooled_draw_max": float(np.max(pooled)),
        "gap_full_n": float(pr_ord_full / pr_sink),
        "gap_matched_n": float(np.mean(matched) / pr_sink),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    logging.info("sink PR                     %.3f  (n=%d, d=%d)", pr_sink, n_sink, dim)
    logging.info("ordinary PR, full n=%-7d %.3f", len(flat), pr_ord_full)
    logging.info("ordinary PR, position-demeaned  %.3f   (gap %.1fx)",
                 pr_ord_demeaned, pr_ord_demeaned / pr_sink)
    logging.info("ordinary PR, matched n=%-4d %.3f  [%.3f, %.3f] over %d draws",
                 n_sink, np.mean(matched), np.min(matched), np.max(matched), args.repeats)
    logging.info("  (pooled-token draw, wrong control, for contrast: %.3f [%.3f, %.3f])",
                 np.mean(pooled), np.min(pooled), np.max(pooled))
    logging.info("gap: %.1fx at full n  ->  %.1fx at matched n",
                 summary["gap_full_n"], summary["gap_matched_n"])
    logging.info("done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
