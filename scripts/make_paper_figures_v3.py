"""Figure for the cross-layer geometry result, both model scales.

Panel (a) plots the scale-invariant on-manifold advantage A = log2(off/on)
against the register participation ratio of the same layer, pooling both
models.  Reporting a ratio rather than a difference in points matters: the
absolute cost of both conditions varies by two orders of magnitude across
depth, because the register's own leading-component energy sets the
perturbation norm and that energy grows sharply when the population becomes
degenerate.

Panel (a) labels only the layers that appear in panel (b).\n\nPanel (b) isolates the three layers at which the two models disagree about
effective dimensionality.  Depth is held fixed within each pair, so if the
advantage follows the participation ratio rather than the depth, dimensionality
and depth are separated.

Reads only the summary JSONs written by layer_participation_sweep.py and
cross_layer_geometry_intervention.py.  No model forward passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1a1a1a"
SMALL_C = "#1b4b8f"
BASE_C = "#c2622a"
POS = "#2c6e49"
NEG = "#a03434"
GREY = "#8a8f99"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def rows(cross_path: Path, sweep_path: Path, label: str) -> list[dict]:
    """One record per non-vacuous layer: PR, on, off, point advantage, log2 ratio."""
    cross = json.loads(cross_path.read_text())
    sweep = json.loads(sweep_path.read_text())
    pr = {e["layer"]: e["register_participation_ratio"] for e in sweep["per_layer"]}
    out = []
    for e in cross["per_layer"]:
        on, off = e["on_manifold_mean"], e["off_manifold_mean"]
        # Layer 11 is vacuous by construction: with only the per-token layer norm
        # after the hook, every condition returns exactly zero.
        if on <= 0.0 or off <= 0.0:
            continue
        adv = e["on_manifold_advantage"]
        out.append({
            "model": label,
            "layer": e["layer"],
            "pr": pr[e["layer"]],
            "on": on,
            "off": off,
            "adv_pts": adv["mean_difference"],
            "ci": (adv["ci95_low"], adv["ci95_high"]),
            "log2": float(np.log2(off / on)),
        })
    return out


def spearman(x, y) -> float:
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return order.astype(float) + 1.0
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/experiments/revision_2026_08"))
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recs = (
        rows(args.root / "cross_layer_small_full/summary.json",
             args.root / "layer_sweep_small/summary.json", "DINOv2-Small")
        + rows(args.root / "cross_layer_base/summary.json",
               args.root / "layer_sweep_base/summary.json", "DINOv2-Base")
    )
    rho = spearman([r["pr"] for r in recs], [r["log2"] for r in recs])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- (a) pooled scatter -------------------------------------------------
    # Log x-axis: ten of the nineteen layers sit between PR 1.00 and 1.25, and a
    # linear axis collapses them into an unreadable column at the left edge.
    ax = axes[0]
    # The observed layers fall into two bands with nothing in between; shading
    # the empty interval makes the band separation, which is the primary
    # statistic, the visual takeaway rather than the rank correlation.
    lo_hi = max(r["pr"] for r in recs if r["pr"] < 1.5)
    hi_lo = min(r["pr"] for r in recs if r["pr"] > 1.5)
    ax.axvspan(lo_hi, hi_lo, color=GREY, alpha=0.13, lw=0, zorder=0)
    ax.text(np.sqrt(lo_hi * hi_lo), -5.4, "no layer\nobserved", ha="center", va="bottom",
            fontsize=6.2, color=GREY)
    ax.axhline(0, color=INK, lw=0.8)
    labelled = {("DINOv2-Small", 0), ("DINOv2-Small", 3), ("DINOv2-Small", 6),
                ("DINOv2-Small", 10), ("DINOv2-Base", 0), ("DINOv2-Base", 3),
                ("DINOv2-Base", 6), ("DINOv2-Base", 10)}
    for label, color, marker in (("DINOv2-Small", SMALL_C, "o"), ("DINOv2-Base", BASE_C, "s")):
        sub = [r for r in recs if r["model"] == label]
        ax.scatter([r["pr"] for r in sub], [r["log2"] for r in sub], c=color, s=27,
                   marker=marker, zorder=3, edgecolors="white", linewidths=0.5, label=label)
        for r in sub:
            if (r["model"], r["layer"]) not in labelled:
                continue
            # Small L10 and Base L3 sit almost on top of each other in the
            # low-PR cluster, so those two labels are placed by hand.
            nudge = {("DINOv2-Small", 10): (-16, 3.5), ("DINOv2-Base", 3): (5, 1.5),
                      ("DINOv2-Small", 6): (-14, -6.0)}
            dx, dy = nudge.get((r["model"], r["layer"]), (5, -1.5))
            ax.annotate(f"L{r['layer']}", (r["pr"], r["log2"]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.0, color=color, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    ax.set_xticklabels(["1.0", "1.25", "1.5", "2.0", "2.5", "3.0"])
    ax.minorticks_off()
    ax.set_xlabel("register participation ratio (log scale)")
    ax.set_ylabel(r"on-manifold advantage  $\log_2(\mathrm{off}/\mathrm{on})$")
    ax.set_title("(a) Advantage falls with effective dimensionality", loc="left")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.4, borderpad=0.2)
    ax.text(0.97, 0.94, rf"Spearman $\rho={rho:.2f}$, $n={len(recs)}$", transform=ax.transAxes,
            fontsize=6.8, color=INK, ha="right")
    ax.text(0.97, 0.86, "removal less disruptive", transform=ax.transAxes,
            fontsize=6.5, color=POS, ha="right")
    ax.text(0.97, 0.05, "removal more disruptive", transform=ax.transAxes,
            fontsize=6.5, color=NEG, ha="right")
    ax.margins(y=0.18)

    # ---- (b) same-depth pairs ----------------------------------------------
    ax = axes[1]
    by = {(r["model"], r["layer"]): r for r in recs}
    pairs = [l for l in (3, 6, 10)
             if ("DINOv2-Small", l) in by and ("DINOv2-Base", l) in by]
    x = np.arange(len(pairs))
    w = 0.36
    ax.axhline(0, color=INK, lw=0.8)
    for off, model, color in ((-w / 2, "DINOv2-Small", SMALL_C), (w / 2, "DINOv2-Base", BASE_C)):
        vals = [by[(model, l)]["log2"] for l in pairs]
        ax.bar(x + off, vals, width=w, color=color, label=model)
        for xi, l in zip(x + off, pairs):
            r = by[(model, l)]
            va = "bottom" if r["log2"] >= 0 else "top"
            dy = 0.12 if r["log2"] >= 0 else -0.12
            ax.text(xi, r["log2"] + dy, f"PR {r['pr']:.2f}", ha="center", va=va,
                    fontsize=5.8, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in pairs])
    ax.set_ylabel(r"$\log_2(\mathrm{off}/\mathrm{on})$")
    ax.set_title("(b) Same depth, different dimensionality", loc="left")
    ax.margins(y=0.30)

    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_sign_inversion.png")
    plt.close(fig)

    # ---- console report, and the LaTeX rows for the appendix table ---------
    print(f"pooled Spearman rho = {rho:.4f}  over n={len(recs)} layers")
    for label in ("DINOv2-Small", "DINOv2-Base"):
        sub = [r for r in recs if r["model"] == label]
        r_one = spearman([r["pr"] for r in sub], [r["log2"] for r in sub])
        print(f"{label:14s} rho = {r_one:+.4f}  (n={len(sub)})")
        pts = spearman([r["pr"] for r in sub], [r["adv_pts"] for r in sub])
        print(f"{'':14s} rho = {pts:+.4f} on the points scale, for comparison")
    print()
    print("LATEX ROWS")
    prev = None
    for r in sorted(recs, key=lambda r: (r["model"] != "DINOv2-Small", r["layer"])):
        name = r["model"] if r["model"] != prev else ""
        prev = r["model"]
        lo, hi = r["ci"]
        print(f"{name} & {r['layer']} & ${r['pr']:.3f}$ & ${r['on']:.3f}$ & ${r['off']:.3f}$ "
              f"& ${r['adv_pts']:+.3f}$ & $[{lo:+.3f}, {hi:+.3f}]$ & ${r['log2']:+.3f}$ \\\\")
    print()
    print("SAME-DEPTH PAIRS")
    for l in pairs:
        s, b = by[("DINOv2-Small", l)], by[("DINOv2-Base", l)]
        print(f"  L{l}: Small PR {s['pr']:.3f} -> A {s['log2']:+.3f} | "
              f"Base PR {b['pr']:.3f} -> A {b['log2']:+.3f} | "
              f"follows PR: {(s['pr'] < b['pr']) == (s['log2'] > b['log2'])}")


if __name__ == "__main__":
    main()
