import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =========================
# Data, read from the evaluate.py results on the combined real-world dataset
# =========================

STRATEGIES = {
    "Top 50 Ctx.": "top50_context",
    "Top 100 Ctx.": "top100_context",
    "Top 200 Ctx.": "top200_context",
    "Top 400 Ctx.": "top400_context",
    "All Ctx.": "all_contexts",
}


def read_results(ordering, strategy):
    with open(f"results_ALL_RL_{ordering}_{strategy}.txt", encoding="utf-8") as f:
        txt = f.read()

    undetected = float(re.search(r"Undetected Ratio: ([\d.]+)", txt).group(1))
    detected_after_5 = float(re.search(r"Detected >5 Ratio: ([\d.]+)", txt).group(1))
    avg_traces = float(re.search(r"Traces to Detection: Avg: ([\d.]+)", txt).group(1))

    return undetected, detected_after_5, avg_traces


strategies = list(STRATEGIES)

random_results = [read_results("RANDOM", s) for s in STRATEGIES.values()]
regr_results = [read_results("LTLTRUST", s) for s in STRATEGIES.values()]

# The undetected ratio does not depend on the ordering
ndt_inf = np.array([r[0] for r in regr_results])

dt_random = np.array([r[1] for r in random_results])
dt_regr = np.array([r[1] for r in regr_results])

avg_random = np.array([r[2] for r in random_results])
avg_regr = np.array([r[2] for r in regr_results])

for name, ndt, dt_rand, dt_reg, avg_rand, avg_reg in zip(strategies, ndt_inf, dt_random, dt_regr, avg_random, avg_regr):
    print(
        f"{name:14s} NDT_inf={ndt:.3f} | DT>5 random={dt_rand:.3f} LTLTrust={dt_reg:.3f} "
        f"| NDT_5 random={ndt + dt_rand:.3f} LTLTrust={ndt + dt_reg:.3f} "
        f"| avg traces random={avg_rand:.2f} LTLTrust={avg_reg:.2f}"
    )

# =========================
# Plot setup
# =========================

x = np.arange(len(strategies))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 6.5))  # keep compact height

base_color = "0.80"
top_color = "0.55"

hatch_random = "//"
hatch_regr = "\\\\"

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

ax.bar(x + width / 2, dt_regr, width,
       bottom=ndt_inf,
       color=top_color, edgecolor="black",
       hatch=hatch_regr, linewidth=0.8)

# =========================
# Labels (sum + avg traces)
# =========================

for i in range(len(strategies)):
    total_random = ndt_inf[i] + dt_random[i]
    total_regr = ndt_inf[i] + dt_regr[i]

    ax.text(
        x[i] - width / 2,
        total_random + 0.01,
        f"{total_random:.3f}\n({avg_random[i]:.2f})",
        ha="center",
        va="bottom",
        fontsize=16,
    )

    ax.text(
        x[i] + width / 2,
        total_regr + 0.01,
        f"{total_regr:.3f}\n({avg_regr[i]:.2f})",
        ha="center",
        va="bottom",
        fontsize=15,
    )

# =========================
# Formatting
# =========================

ax.set_xticks(x)
ax.set_xticklabels(strategies, rotation=60, ha="right", fontsize=20)

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
regr_patch = mpatches.Patch(
    facecolor=top_color,
    hatch=hatch_regr,
    edgecolor="black",
    label=r"$DT_{>5}$ (LTLTrust Ordering)",
)

avg_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="None",
    label="Top label = NDT + DT; parentheses = avg. traces"
)

ax.legend(handles=[base_patch, random_patch, regr_patch], fontsize=18)

# =========================
# Limits
# =========================

ax.set_ylim(0, np.max(ndt_inf + np.maximum(dt_random, dt_regr)) * 1.35)

plt.tight_layout()

# =========================
# Save SVG
# =========================

plt.savefig("stacked_bar_plot.svg", format="svg", bbox_inches="tight")