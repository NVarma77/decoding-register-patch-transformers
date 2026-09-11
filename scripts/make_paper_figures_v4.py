"""Figure for the pooling-artifact result.

Panel (a): the register participation ratio across depth, computed three ways.
Pooled over all four register slots -- the statistic the geometric account rests
on -- sits near one through the middle layers, while the same quantity computed
within each slot, or after subtracting each slot's own mean, is an order of
magnitude larger.  The pooled number is measuring the offset between slots, not
the dimensionality of the population any single intervention acts on.

Panel (b): the consequence.  The leading-PC advantage A = log2(rand/PC) falls
with pooled participation ratio, which is the relationship a geometric account
predicts, and rises with the per-slot participation ratio.  The sign of the
relationship depends on which of the two is used, so the pooled version does not
establish the account.

Reads only summary JSONs written by the experiment scripts.
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
POOL_C = "#a03434"
GREY = "#8a8f99"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK,
    "axes.linewidth": 0.7, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK, "figure.dpi": 400,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def spearman(x, y) -> float:
    rank = lambda v: (np.argsort(np.argsort(np.asarray(v, float))) + 1.0)
    a, b = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def load(root: Path, cross: str, slots: str, label: str):
    c = json.loads((root / cross / "summary.json").read_text())
    g = json.loads((root / slots / "summary.json").read_text())
    per = {e["layer"]: e for e in g["per_layer"]}
    rows = []
    for e in c["per_layer"]:
        on, off = e["on_manifold_mean"], e["off_manifold_mean"]
        if on <= 0 or off <= 0:
            continue
        q = per[e["layer"]]
        rows.append({"model": label, "layer": e["layer"],
                     "A": float(np.log2(off / on)),
                     "pooled": q["pooled_pr"],
                     "slot": float(np.mean(q["per_slot_pr"])),
                     "dem": q["demeaned_pr"]})
    return rows, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/experiments/revision_2026_08"))
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    s_rows, s_geo = load(args.root, "cross_layer_small_full", "per_slot_small", "DINOv2-Small")
    b_rows, b_geo = load(args.root, "cross_layer_base", "per_slot_base", "DINOv2-Base")
    rows = s_rows + b_rows

    null = json.loads((args.root / "resample_ablation_null" / "summary.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3),
                             gridspec_kw={"width_ratios": [1.10, 0.95, 1.05]})

    # ---- (a) three ways of measuring the same population --------------------
    ax = axes[0]
    for geo, label, style, fill in ((s_geo, "Small", "-", True), (b_geo, "Base", "--", False)):
        L = [e["layer"] for e in geo["per_layer"]]
        for key, colour, mark in (("pooled", POOL_C, "o"), ("slot", SMALL_C, "s")):
            y = ([e["pooled_pr"] for e in geo["per_layer"]] if key == "pooled"
                 else [np.mean(e["per_slot_pr"]) for e in geo["per_layer"]])
            name = "pooled over slots" if key == "pooled" else "within slot"
            ax.plot(L, y, style, color=colour, lw=1.3, marker=mark, ms=3.0,
                    markerfacecolor=(colour if fill else "white"),
                    markeredgecolor=colour, markeredgewidth=0.8,
                    label=(name if label == "Small" else None))
    ax.axhline(1.0, color=GREY, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("register participation ratio")
    ax.set_title("(a) Pooled statistic is the outlier", loc="left")
    ax.set_xticks(range(0, 12, 2))
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.9,
              borderpad=0.1, labelspacing=0.25, fontsize=6.4)
    ax.set_ylim(0.7, 400)

    # ---- (b) the relationship changes sign with the definition --------------
    ax = axes[1]
    ax.axhline(0, color=INK, lw=0.8)
    r_pool = spearman([r["pooled"] for r in rows], [r["A"] for r in rows])
    r_slot = spearman([r["slot"] for r in rows], [r["A"] for r in rows])
    ax.scatter([r["pooled"] for r in rows], [r["A"] for r in rows], s=22,
               c=POOL_C, marker="o", edgecolors="white", linewidths=0.4,
               label=rf"pooled  ($\rho={r_pool:+.2f}$)", zorder=3)
    ax.scatter([r["slot"] for r in rows], [r["A"] for r in rows], s=22,
               c=SMALL_C, marker="s", edgecolors="white", linewidths=0.4,
               label=rf"within slot  ($\rho={r_slot:+.2f}$)", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("register participation ratio (log scale)")
    ax.set_ylabel(r"leading-PC advantage  $\log_2(\mathrm{rand}/\mathrm{PC})$")
    ax.set_title("(b) Sign depends on the definition", loc="left")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.3, borderpad=0.2)
    ax.margins(y=0.16)

    # ---- (c) no null matches both magnitude and support ---------------------
    ax = axes[2]
    cm = null["condition_means"]
    spread = null["swap_delta_norm_mean"]
    treat = null["treatment_norm_mean"]

    ax.axvspan(spread * 0.55, spread * 1.8, color=SMALL_C, alpha=0.10, lw=0)
    ax.axvline(treat, color=GREY, lw=0.7, ls="--")

    pts = [
        (spread, cm["swap_slot"], "swap (on support)", SMALL_C, "o"),
        (treat, cm["clean_top5"], "top-five removal", INK, "*"),
        (treat, cm["rescaled_slot"], "swap, rescaled", POOL_C, "s"),
        (treat, cm["random_span"], "random span", POOL_C, "^"),
    ]
    for x, y, lab, c, m in pts:
        ax.plot([x], [y], m, color=c, ms=(8.5 if m == "*" else 4.4),
                markeredgecolor="white", markeredgewidth=0.4, zorder=3, label=lab)

    ax.annotate("", xy=(spread, 14.6), xytext=(treat, 14.6),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    ax.text(np.sqrt(spread * treat), 15.0,
            r"$%.0f\times$" % null["rescaling_amplification_mean"],
            ha="center", va="bottom", fontsize=7)
    ax.text(spread, 19.0, "population's\nown spread", ha="center", va="top",
            fontsize=6.2, color=SMALL_C, linespacing=0.95)
    ax.text(treat, 19.0, "treatment\nmagnitude", ha="center", va="top",
            fontsize=6.2, color=GREY, linespacing=0.95)

    ax.set_xscale("log")
    ax.set_xlim(2.0, 400)
    ax.set_ylim(-1.5, 23.0)
    ax.set_xlabel("perturbation norm (log scale)")
    ax.set_ylabel("cost (points)")
    ax.set_title("(c) No null matches both", loc="left")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2,
              borderpad=0.1, labelspacing=0.2, fontsize=6.0)

    fig.tight_layout(w_pad=1.1)
    fig.savefig(args.output_dir / "figure_pooling_artifact.png")
    plt.close(fig)

    print(f"n = {len(rows)} layers")
    print(f"rho(A, pooled PR)     = {r_pool:+.4f}")
    print(f"rho(A, within-slot PR)= {r_slot:+.4f}")
    print(f"rho(A, demeaned PR)   = {spearman([r['dem'] for r in rows], [r['A'] for r in rows]):+.4f}")
    print()
    print("LATEX ROWS (layer, pooled, per-slot list, demeaned)")
    for geo, name in ((s_geo, "DINOv2-Small"), (b_geo, "DINOv2-Base")):
        first = True
        for e in geo["per_layer"]:
            if e["layer"] == 11:
                continue
            slots = ", ".join("%.1f" % v for v in e["per_slot_pr"])
            print(f"{name if first else ''} & {e['layer']} & ${e['pooled_pr']:.3f}$ & "
                  f"${slots}$ & ${e['demeaned_pr']:.1f}$ \\\\")
            first = False


if __name__ == "__main__":
    main()
