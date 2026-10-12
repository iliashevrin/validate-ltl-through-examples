#!/usr/bin/env python3
"""
Acceptance of the generated traces (accepting / rejecting / inconclusive)
with respect to the candidate formula and with respect to the ground truth,
over all traces generated for a set of Cand-GT pairs and over the traces that
distinguish the candidate from the ground truth, regardless of the order in
which they would be examined. The ground truth's acceptance is the one an
engineer bases their evaluation on.
"""
import argparse
import csv
from collections import Counter

from mutation_based import generate_traces
from utils import check_acceptance, translate


LABELS = [(True, "Accepting"), (False, "Rejecting"), (None, "Inconclusive")]


def print_table(title, generated, detecting):
    total_generated = sum(generated.values())
    total_detecting = sum(detecting.values())

    print(f"{title:<22}{'Generated':>18}{'Detecting':>18}{'Detecting / generated':>24}")

    for value, label in LABELS:
        print(
            f"{label:<22}"
            f"{generated[value]:>9} ({generated[value] / total_generated:5.1%})"
            f"{detecting[value]:>9} ({detecting[value] / total_detecting:5.1%})"
            f"{detecting[value] / generated[value]:>24.1%}"
        )

    print(f"{'Total':<22}{total_generated:>18}{total_detecting:>18}{total_detecting / total_generated:>24.1%}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="?", default="ALL_RL/pairs_claude.csv")
    parser.add_argument("--strategy", default="all_contexts")

    args = parser.parse_args()

    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    generated = {"candidate": Counter(), "ground_truth": Counter()}
    detecting = {"candidate": Counter(), "ground_truth": Counter()}

    for pair in pairs:
        gt_aut = translate(pair["Ground Truth"])

        for trace, candidate_acceptance, _ in generate_traces(pair["Response"], args.strategy):
            gt_acceptance = check_acceptance(gt_aut, trace)

            generated["candidate"][candidate_acceptance] += 1
            generated["ground_truth"][gt_acceptance] += 1

            # The engineer detects the mismatch, as in utils.simulate_user
            if gt_acceptance != candidate_acceptance:
                detecting["candidate"][candidate_acceptance] += 1
                detecting["ground_truth"][gt_acceptance] += 1

    print(f"{len(pairs)} Cand-GT pairs from {args.pairs} ({args.strategy})\n")
    print_table("Candidate acceptance", generated["candidate"], detecting["candidate"])
    print_table("GT acceptance", generated["ground_truth"], detecting["ground_truth"])


if __name__ == "__main__":
    main()
