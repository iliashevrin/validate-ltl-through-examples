import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Plot NDT_5 of the trace generation strategies.")
parser.add_argument("--output", default="stacked_bar_plot.svg")
parser.add_argument("--results-dir", default=".", help="Directory of the evaluate.py results files")
args = parser.parse_args()

# =========================
# Data, read from the evaluate.py results on the combined real-world dataset
# =========================

STRATEGIES = {
    "Top 10 Ctx.": "top10_context",
    "Top 25 Ctx.": "top25_context",
    "Top 50 Ctx.": "top50_context",
    "Top 100 Ctx.": "top100_context",
    "Top 200 Ctx.": "top200_context",
    "All Ctx.": "all_contexts",
}


def read_results(ordering, strategy):
    with open(os.path.join(args.results_dir, f"results_ALL_RL_{ordering}_{strategy}.txt"), encoding="utf-8") as f:
        txt = f.read()

    undetected = float(re.search(r"Undetected Ratio: ([\d.]+)", txt).group(1))
    detected_after_5 = float(re.search(r"Detected >5 Ratio: ([\d.]+)", txt).group(1))
    avg_traces = float(re.search(r"Traces to Detection: Avg: ([\d.]+)", txt).group(1))

    return undetected, detected_after_5, avg_traces


strategies = list(STRATEGIES)

random_results = [read_results("RANDOM", s) for s in STRATEGIES.values()]
ltltrust_results = [read_results("LTLTRUST", s) for s in STRATEGIES.values()]

# The undetected ratio does not depend on the ordering
ndt_inf = np.array([r[0] for r in ltltrust_results])

dt_random = np.array([r[1] for r in random_results])
dt_ltltrust = np.array([r[1] for r in ltltrust_results])

avg_random = np.array([r[2] for r in random_results])
avg_ltltrust = np.array([r[2] for r in ltltrust_results])

for name, ndt, dt_rand, dt_lt, avg_rand, avg_lt in zip(strategies, ndt_inf, dt_random, dt_ltltrust, avg_random, avg_ltltrust):
    print(
        f"{name:14s} NDT_inf={ndt:.3f} | DT>5 random={dt_rand:.3f} LTLTrust={dt_lt:.3f} "
        f"| NDT_5 random={ndt + dt_rand:.3f} LTLTrust={ndt + dt_lt:.3f} "
        f"| avg traces random={avg_rand:.2f} LTLTrust={avg_lt:.2f}"
    )

# =========================
# Plot setup
# =========================

x = np.arange(len(strategies))
# Sized for a one-column paper, where the figure spans most of the text width:
# wide bars and fonts that stay legible at about half the figure's size
width = 0.44

fig, ax = plt.subplots(figsize=(10, 4.2))  # keep compact height

base_color = "0.80"
top_color = "0.55"

hatch_random = "//"
hatch_ltltrust = "\\\\"

# =========================
# Bars
# =========================

ax.bar(x - width / 2, ndt_inf, width,
       color=base_color, edgecolor="black", linewidth=0.8)

ax.bar(x - width / 2, dt_random, width,
       bottom=ndt_inf,
       color=top_color, edgecolor="black",
       hatch=hatch_random, linewidth=0.8)

ax.bar(x + width / 2, ndt_inf, width,
       color=base_color, edgecolor="black", linewidth=0.8)

ax.bar(x + width / 2, dt_ltltrust, width,
       bottom=ndt_inf,
       color=top_color, edgecolor="black",
       hatch=hatch_ltltrust, linewidth=0.8)

# =========================
# Labels (sum + avg traces)
# =========================

for i in range(len(strategies)):
    total_random = ndt_inf[i] + dt_random[i]
    total_ltltrust = ndt_inf[i] + dt_ltltrust[i]

    ax.text(
        x[i] - width / 2,
        total_random + 0.01,
        f"{total_random:.3f}\n({avg_random[i]:.2f})",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    ax.text(
        x[i] + width / 2,
        total_ltltrust + 0.01,
        f"{total_ltltrust:.3f}\n({avg_ltltrust[i]:.2f})",
        ha="center",
        va="bottom",
        fontsize=14,
    )

# =========================
# Formatting
# =========================

ax.set_xticks(x)
# Horizontal labels over two lines, e.g. "Top 25" above "Ctx."
ax.set_xticklabels([s.replace(" Ctx.", "\nCtx.") for s in strategies], fontsize=17)
ax.tick_params(axis="y", labelsize=13)
# Little empty space beside the outer bars
ax.margins(x=0.02)

ax.set_xlabel("")
ax.set_ylabel("")

# =========================
# Legend
# =========================

base_patch = mpatches.Patch(color=base_color, label=r"$NDT_\infty$")
random_patch = mpatches.Patch(
    facecolor=top_color,
    hatch=hatch_random,
    edgecolor="black",
    label=r"$DT_{>5}$ (Random)",
)
ltltrust_patch = mpatches.Patch(
    facecolor=top_color,
    hatch=hatch_ltltrust,
    edgecolor="black",
    label=r"$DT_{>5}$ (LTLTrust Ordering)",
)

avg_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="None",
    label="Top label = NDT + DT; parentheses = avg. traces"
)

# One row above the axes, so the bars need no headroom for the legend
ax.legend(
    handles=[base_patch, random_patch, ltltrust_patch],
    fontsize=16,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    frameon=False,
    handlelength=1.5,
    columnspacing=1.2,
)

# =========================
# Limits
# =========================

# Leave room above the bars for their labels
ax.set_ylim(0, np.max(ndt_inf + np.maximum(dt_random, dt_ltltrust)) * 1.4)

plt.tight_layout()

# =========================
# Save SVG
# =========================

plt.savefig(args.output, format="svg", bbox_inches="tight")