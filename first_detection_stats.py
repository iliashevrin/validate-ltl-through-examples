#!/usr/bin/env python3
"""
What the first trace to distinguish a Cand-GT pair looks like under each
ordering: its share of accepting, rejecting and inconclusive traces with respect
to the ground truth, which is the evaluation the engineer has to make.

The random ordering is exact: the first distinguishing trace of a random order is
uniform over the pair's distinguishing traces. The last row is the share among all
distinguishing traces, as trace_acceptance_stats.py reports it. It weighs each pair by
its number of distinguishing traces, whereas the rows above weigh the pairs equally.
"""
import argparse
import csv
from collections import Counter

from evaluate import by_length, ltltrust, static_sa_usefulness, static_usefulness
from mutation_based import generate_traces
from utils import check_acceptance, translate


ORDERINGS = [
    ("Shortest First", by_length),
    ("LTLTrust", static_usefulness),
    ("SA LTLTrust", static_sa_usefulness),
    ("Reg LTLTrust", ltltrust),
]
LABELS = [(True, "Accepting"), (False, "Rejecting"), (None, "Inconclusive")]


def shares(counts):
    total = sum(counts[value] for value, _ in LABELS)
    return "".join(f"{counts[value] / total:>14.1%}" for value, _ in LABELS) + f"{total:>12.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="?", default="ALL_RL/pairs_claude.csv")
    parser.add_argument("--strategy", default="all_contexts")
    args = parser.parse_args()

    first = {name: Counter() for name, _ in ORDERINGS}
    uniform = Counter()
    overall = Counter()

    with open(args.pairs, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            traces = generate_traces(row["Response"], args.strategy)
            gt = translate(row["Ground Truth"])
            verdict = {t: check_acceptance(gt, t) for t, _, _ in traces}
            detecting = [t for t, acc, _ in traces if verdict[t] != acc]

            if not detecting:
                continue

            for t in detecting:
                uniform[verdict[t]] += 1 / len(detecting)
                overall[verdict[t]] += 1

            for name, order in ORDERINGS:
                ordered = order(row["Response"], list(traces), args.strategy)
                first[name][next(verdict[t] for t, acc, _ in ordered if verdict[t] != acc)] += 1

    print(f"{'First distinguishing trace':<30}" + "".join(f"{label:>14}" for _, label in LABELS) + f"{'Total':>12}")
    print(f"{'Random':<30}{shares(uniform)}")
    for name, _ in ORDERINGS:
        print(f"{name:<30}{shares(first[name])}")
    print(f"{'All distinguishing traces':<30}{shares(overall)}")


if __name__ == "__main__":
    main()
