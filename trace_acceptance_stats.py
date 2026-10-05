#!/usr/bin/env python3
"""
Acceptance of the generated traces with respect to the candidate formula
(accepting / rejecting / inconclusive), over all traces generated for a set
of Cand-GT pairs and over the traces that distinguish the candidate from the
ground truth, regardless of the order in which they would be examined.
"""
import argparse
import csv
from collections import Counter

from mutation_based import generate_traces
from utils import simulate_user


LABELS = [(True, "Accepting"), (False, "Rejecting"), (None, "Inconclusive")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="?", default="ALL_RL/pairs_claude.csv")
    parser.add_argument("--strategy", default="all_contexts")

    args = parser.parse_args()

    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    generated = Counter()
    detecting = Counter()

    for pair in pairs:
        for trace, candidate_acceptance, _ in generate_traces(pair["Response"], args.strategy):
            generated[candidate_acceptance] += 1

            if simulate_user(pair["Ground Truth"], trace, candidate_acceptance):
                detecting[candidate_acceptance] += 1

    total_generated = sum(generated.values())
    total_detecting = sum(detecting.values())

    print(f"{len(pairs)} Cand-GT pairs from {args.pairs} ({args.strategy})\n")
    print(f"{'Candidate acceptance':<22}{'Generated':>18}{'Detecting':>18}{'Detecting / generated':>24}")

    for value, label in LABELS:
        print(
            f"{label:<22}"
            f"{generated[value]:>9} ({generated[value] / total_generated:5.1%})"
            f"{detecting[value]:>9} ({detecting[value] / total_detecting:5.1%})"
            f"{detecting[value] / generated[value]:>24.1%}"
        )

    print(f"{'Total':<22}{total_generated:>18}{total_detecting:>18}{total_detecting / total_generated:>24.1%}")


if __name__ == "__main__":
    main()
