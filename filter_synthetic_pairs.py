#!/usr/bin/env python3
"""
Exclude synthetic Cand-GT pairs whose candidate uses atomic propositions
outside the AP set given to the model, matching the exclusion applied to the
real-world datasets in generate_gt_pairs.py, and report the per-source
counts behind Table I.

An AP counts as provided when it appears in the tuple's AP list or in the
ground truth itself.
"""
import argparse
import ast
import csv
from collections import Counter

import pandas as pd
import spot
spot.setup()


SOURCE_NAMES = {
    "NL-to-LTL-Synthetic-Datasetprocessed_data": "Synthetic",
    "lifted_data_pureLTL.jsonl": "NL2TL",
    "VLTL-Bench_New": "VLTL (New)",
    "VLTL-Bench_Prev": "VLTL (Prev)",
    "ConformalNL2LTL": "Conformal",
}


def aps(formula):
    return {str(ap) for ap in spot.atomic_prop_collect(spot.formula(formula))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data_collection/data_all.csv")
    parser.add_argument("--tuples", default="data_collection/final_df_downsampled.csv")
    parser.add_argument("--output", default="data_collection/data_all_in_ap_set.csv")

    args = parser.parse_args()

    tuples = pd.read_csv(args.tuples, sep=";")

    # Ground truths appear in the pairs either as written in the dataset or as printed by Spot
    by_gt = {}
    for _, row in tuples.iterrows():
        info = (row["source"], set(ast.literal_eval(row["APs"])))
        by_gt.setdefault(str(row["original LTL"]).strip(), info)
        by_gt.setdefault(str(row["Spot LTL"]).strip(), info)

    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    kept = []
    mismatches = Counter()
    excluded = Counter()
    unmatched = 0

    for pair in pairs:
        ground_truth = pair["Ground Truth"].strip()

        if ground_truth not in by_gt:
            unmatched += 1
            kept.append(pair)
            continue

        source, provided = by_gt[ground_truth]
        mismatches[source] += 1

        if aps(pair["Response"]) <= provided | aps(ground_truth):
            kept.append(pair)
        else:
            excluded[source] += 1

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Ground Truth", "Response"])
        writer.writeheader()
        writer.writerows(kept)

    totals = tuples["source"].value_counts()

    print(f"{'Dataset':<14}{'Tuples':>8}{'Excluded':>10}{'Total Cand.':>13}{'Mismatches':>12}{'Accuracy':>10}")

    for source, name in SOURCE_NAMES.items():
        total = totals[source] - excluded[source]
        mismatch = mismatches[source] - excluded[source]
        print(f"{name:<14}{totals[source]:>8}{excluded[source]:>10}{total:>13}{mismatch:>12}{1 - mismatch / total:>10.3f}")

    total = len(tuples) - sum(excluded.values())
    mismatch = sum(mismatches.values()) - sum(excluded.values())
    print(f"{'Total':<14}{len(tuples):>8}{sum(excluded.values()):>10}{total:>13}{mismatch:>12}{1 - mismatch / total:>10.3f}")

    print(f"\nKept {len(kept)} of {len(pairs)} pairs in {args.output}; {unmatched} pairs had no matching tuple")


if __name__ == "__main__":
    main()
