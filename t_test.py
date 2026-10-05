import argparse
import itertools
import re

import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon


DEFAULT_RESULTS = [
    "Random=results_ALL_RL_RANDOM_all_contexts.txt",
    "Shortest First=results_ALL_RL_BY_LENGTH_all_contexts.txt",
    "LTLTrust=results_ALL_RL_LTLTRUST_all_contexts.txt",
    "SA LTLTrust=results_ALL_RL_LTLTRUST_PLUS_all_contexts.txt",
]


def traces_to_detection(results_file):
    """
    The per-pair number of traces until mismatch detection, as listed at the
    end of a results file written by evaluate.py (averaged over the repeats
    for the random ordering). Undetected pairs are not listed, and they are
    the same for every ordering, so the lists of two orderings are paired.
    """
    with open(results_file, encoding="utf-8") as f:
        match = re.search(r"Traces to Detection List: \[(.*)\]", f.read())

    return np.array([float(x) for x in re.findall(r"'([\d.]+)'", match.group(1))])


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the original order."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0

    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)

    return adjusted


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise paired comparison of the traces needed for mismatch detection by several orderings."
    )
    parser.add_argument(
        "results",
        nargs="*",
        default=DEFAULT_RESULTS,
        metavar="[NAME=]RESULTS_FILE",
        help="evaluate.py results files to compare pairwise, default: the four orderings on ALL_RL",
    )

    args = parser.parse_args()

    orderings = {}
    for item in args.results:
        name, _, path = item.rpartition("=")
        orderings[name or path] = traces_to_detection(path)

    comparisons = []

    for (name_a, a), (name_b, b) in itertools.combinations(orderings.items(), 2):
        if len(a) != len(b):
            raise ValueError(f"{name_a} and {name_b} list different numbers of detected pairs")

        diff = a - b
        comparisons.append({
            "pair": f"{name_a} vs {name_b}",
            "n": len(diff),
            "mean_diff": diff.mean(),
            "dz": diff.mean() / diff.std(ddof=1),
            "t": ttest_rel(a, b),
            "w": wilcoxon(a, b),
            "nonzero": int(np.count_nonzero(diff)),
        })

    t_adjusted = holm(np.array([c["t"].pvalue for c in comparisons]))
    w_adjusted = holm(np.array([c["w"].pvalue for c in comparisons]))

    print(f"{'Comparison (A vs B)':<32}{'n':>5}{'mean A-B':>10}{'d_z':>7}{'t':>8}"
          f"{'p (t, Holm)':>14}{'p (Wilcoxon, Holm)':>21}{'non-zero':>10}")

    for c, p_t, p_w in zip(comparisons, t_adjusted, w_adjusted):
        print(f"{c['pair']:<32}{c['n']:>5}{c['mean_diff']:>10.2f}{c['dz']:>7.2f}{c['t'].statistic:>8.2f}"
              f"{p_t:>14.2e}{p_w:>21.2e}{c['nonzero']:>10}")

    print("\nTwo-sided tests; d_z is the mean paired difference over its standard deviation;")
    print("the Wilcoxon signed-rank test discards the pairs with equal numbers of traces (zero differences).")


if __name__ == "__main__":
    main()
