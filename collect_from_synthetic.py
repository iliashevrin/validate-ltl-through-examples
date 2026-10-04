#!/usr/bin/env python3
import argparse
import csv
import random
import sys

from datasets import load_dataset, Dataset

import spot
spot.setup()


DATASET_NAME = "cRick/NL-to-LTL-Synthetic-Dataset"


def extract_ap_set(ltl_formula: str) -> str:
    f = spot.formula(ltl_formula)
    aps = sorted(str(ap) for ap in spot.atomic_prop_collect(f))
    return "{" + " ".join(aps) + "}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_csv")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset = load_dataset(
        DATASET_NAME,
        split=args.split,
        streaming=False,
    )

    if not hasattr(dataset, "__len__"):
        dataset = Dataset.from_list(list(dataset))

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.n)

    rows = []
    skipped = 0

    for idx in indices:
        row = dataset[int(idx)]

        en = row["en"]
        ltl = row["ltl"]

        try:
            ap_set = extract_ap_set(ltl)
        except Exception as exc:
            print(
                f"Skipping row due to Spot parse error:\n"
                f"  LTL: {ltl}\n"
                f"  Error: {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        rows.append(
            {
                "Natural Language": en,
                "Ground Truth": ltl,
                "Atomic Proposition": ap_set,
            }
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Natural Language", "Ground Truth", "Atomic Proposition"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Skipped {skipped} rows due to parse errors")


if __name__ == "__main__":
    main()