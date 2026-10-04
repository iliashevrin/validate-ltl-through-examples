import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (results file, line style, line width), styled as described in the paper's caption:
# dashed: random, dotted: shortest-first, solid: LTLTrust, bold: SA LTLTrust
ORDERINGS = [
    ("results_ALL_RL_RANDOM_all_contexts.txt", (0, (8, 4)), 1.5),
    ("results_ALL_RL_BY_LENGTH_all_contexts.txt", (0, (2, 1.5)), 1.5),
    ("results_ALL_RL_LTLTRUST_all_contexts.txt", "solid", 1.5),
    ("results_ALL_RL_LTLTRUST_PLUS_all_contexts.txt", "solid", 4),
]


def confidence_scores(results_file):
    with open(results_file, encoding="utf-8") as f:
        scores = re.findall(r"Confidence score for n=(\d+): ([\d.]+)", f.read())

    return [int(n) for n, _ in scores], [float(score) for _, score in scores]


def main():
    parser = argparse.ArgumentParser(description="Plot conf_n for n=0..15 per trace ordering.")
    parser.add_argument("--output", default="chart.svg")

    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.0, 3.71))

    for results_file, style, width in ORDERINGS:
        n, conf = confidence_scores(results_file)
        ax.plot(n, conf, color="0.6", linestyle=style, linewidth=width)

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.grid(color="0.8", linewidth=0.8)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, format="svg")


if __name__ == "__main__":
    main()
