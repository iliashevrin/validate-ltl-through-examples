#!/usr/bin/env python3
"""
Count the LLM responses excluded from the Cand-GT collection, i.e. responses
that failed syntactic validation or used atomic propositions outside the
provided AP set, based on the responses CSVs written by generate_gt_pairs.py.

Counts are over unique Cand-GT pairs, matching how generate_gt_pairs.py
computes the total number of candidates and the accuracy.
"""
import argparse
import csv


DEFAULT_RESPONSES = {
    "Dwyer": "Dwyer/responses_claude.csv",
    "SpaceWire": "SpaceWire/responses_claude.csv",
    "Textbook": "textbook/responses_claude.csv",
    "ARTEMIS": "ARTEMIS/responses_claude.csv",
}

STATUSES = ["equivalent", "mismatch", "syntax_error", "ap_outside_set", "compare_error"]


def count(path, include_duplicates):

    counts = {status: 0 for status in STATUSES}

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Duplicate"] == "True" and not include_duplicates:
                continue

            counts[row["Status"]] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        nargs="+",
        metavar="NAME=CSV",
        help="Responses CSVs to count, default: the four real-world datasets",
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Count every response instead of unique Cand-GT pairs",
    )

    args = parser.parse_args()

    if args.responses:
        datasets = dict(item.split("=", 1) for item in args.responses)
    else:
        datasets = DEFAULT_RESPONSES

    header = (
        f"{'Dataset':<12}{'Responses':>10}{'Syntax':>8}{'AP set':>8}{'Excluded':>10}"
        f"{'Total Cand.':>13}{'Mismatches':>12}{'Accuracy':>10}"
    )
    print(header)
    print("-" * len(header))

    totals = {status: 0 for status in STATUSES}

    def report(name, counts):
        responses = sum(counts.values())
        excluded = counts["syntax_error"] + counts["ap_outside_set"]
        candidates = counts["equivalent"] + counts["mismatch"]
        accuracy = counts["equivalent"] / candidates if candidates else 0.0

        print(
            f"{name:<12}{responses:>10}{counts['syntax_error']:>8}{counts['ap_outside_set']:>8}"
            f"{excluded:>10}{candidates:>13}{counts['mismatch']:>12}{accuracy:>10.3f}"
        )

        if counts["compare_error"]:
            print(f"{'':<12}(plus {counts['compare_error']} responses Spot could not compare)")

    for name, path in datasets.items():
        counts = count(path, args.include_duplicates)
        report(name, counts)

        for status in STATUSES:
            totals[status] += counts[status]

    print("-" * len(header))
    report("Total", totals)

    responses = sum(totals.values())
    excluded = totals["syntax_error"] + totals["ap_outside_set"]
    print(
        f"\nExcluded {excluded} of {responses} responses ({excluded / responses:.1%}): "
        f"{totals['syntax_error']} failed syntactic validation ({totals['syntax_error'] / responses:.1%}), "
        f"{totals['ap_outside_set']} used APs outside the provided set ({totals['ap_outside_set'] / responses:.1%})"
    )


if __name__ == "__main__":
    main()
