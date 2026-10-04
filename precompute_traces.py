#!/usr/bin/env python3
"""
Generate the traces of every candidate formula once, in a single process and
a fixed order, and store them in the trace cache read by generate_traces.

Spot orders formula operands by creation order within a process, so traces
generated on the fly depend on what the process handled before; with the
cache, training, all trace orderings and the context ranking use identical
traces for each candidate.
"""
import argparse
import csv
import json
import time

from mutation_based import generate_raw_traces, TRACE_CACHE


DEFAULT_PAIRS = [
    "data_collection/data_all_in_ap_set.csv",
    "ALL_RL/pairs_claude.csv",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="*", default=DEFAULT_PAIRS, help="Cand-GT pair CSVs")
    parser.add_argument("--output", default=TRACE_CACHE)

    args = parser.parse_args()

    candidates = []
    seen = set()

    for path in args.pairs:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Response"] not in seen:
                    seen.add(row["Response"])
                    candidates.append(row["Response"])

    print(f"{len(candidates)} distinct candidates from {', '.join(args.pairs)}", flush=True)

    cache = {}
    start = time.time()

    for index, candidate in enumerate(candidates, start=1):
        try:
            cache[candidate] = generate_raw_traces(candidate)
        except Exception as e:
            print(f"[WARNING] Skipping candidate {candidate}: {e}", flush=True)

        if index % 500 == 0:
            print(f"{index}/{len(candidates)} candidates, {time.time() - start:.0f}s", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"Wrote traces of {len(cache)} candidates to {args.output}", flush=True)


if __name__ == "__main__":
    main()
