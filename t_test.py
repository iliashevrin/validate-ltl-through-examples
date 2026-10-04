import argparse
import re

import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon


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


def t_test(A, B):

    diff = A - B

    print("Sizes:", len(A), len(B))

    print("Mean diff:", np.mean(diff))
    print("Median diff:", np.median(diff))
    print("Std diff:", np.std(diff, ddof=1))

    # Paired t-test
    result = ttest_rel(A, B)

    print("t-statistic:", result.statistic)
    print("p-value:", result.pvalue)

    res = wilcoxon(A, B, alternative="less")
    print("Wilcoxon p-value:", res.pvalue)


def main():
    parser = argparse.ArgumentParser(
        description="Paired comparison of the traces needed for mismatch detection by two orderings."
    )
    parser.add_argument("results_a", help="e.g. results_ALL_RL_LTLTRUST_all_contexts.txt")
    parser.add_argument("results_b", help="e.g. results_ALL_RL_RANDOM_all_contexts.txt")

    args = parser.parse_args()

    print(f"Comparing {args.results_a} with {args.results_b}")
    t_test(traces_to_detection(args.results_a), traces_to_detection(args.results_b))


if __name__ == "__main__":
    main()
