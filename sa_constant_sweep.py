#!/usr/bin/env python3
"""
The normalization constants of the SA usefulness score, for the SA_LTLTRUST
ordering: for every pair of constants, ranks the mutation contexts by their mean
SA usefulness over the training pairs and reports the trace length, the literal
count, the traces to detection and the confidence, as evaluate.py does, on the
held-out pairs, where the constants are chosen, and on the real-world pairs.

With --validation, the constants are instead chosen on a validation split of the
training pairs (a fifth of them, by formula), with the contexts ranked on the rest,
so that the held-out pairs remain a pure test set.

Infinite constants drop a term of the score: both infinite is plain usefulness,
that is, the LTLTRUST ordering. The pair (6, 5), the default, must reproduce results_*_SA_LTLTRUST_*.
"""
import argparse
import csv
import math
import statistics

import pandas as pd

from evaluate import DATASIZE
from mutation_based import generate_traces
from train_ordering import build_training_data, smoothed_ratio_plus, split_by_formula, static_trace_ranking

# The synthetic datasets' accuracy (Table 2), for the confidence on the validation split
SYNTHETIC_ACCURACY = 0.433
from utils import check_acceptance, count_literals, trace_len, translate


LEN_REFS = [1.5, 3, 6, math.inf]
LIT_REFS = [2.5, 5, 10, math.inf]
STRATEGY = "all_contexts"


def pair_data(gt_formula, candidate):
    traces = generate_traces(candidate, STRATEGY)
    gt = translate(gt_formula)
    return {"traces": traces, "detects": {(t, acc): check_acceptance(gt, t) != acc for t, acc, _ in traces}}


def load_pairs(path, dataset):
    with open(path, newline="", encoding="utf-8") as f:
        pairs = [pair_data(row["Ground Truth"], row["Response"]) for row in csv.DictReader(f)]
    return {"pairs": pairs, "accuracy": 1 - len(pairs) / DATASIZE[dataset]}


def measure(data, scores):
    seen_detected, lengths, literals = [], [], []
    undetected_within = {n: 0 for n in (1, 5, 10)}

    for pair in data["pairs"]:
        seen, length, lit, detected = 0, 0, 0, False
        for trace, acc, _ in static_trace_ranking(list(pair["traces"]), scores):
            seen += 1
            length += trace_len(trace)
            lit += count_literals(trace)
            if pair["detects"][(trace, acc)]:
                detected = True
                break
        lengths.append(length / seen if seen else 0)
        literals.append(lit / seen if lit else 0)
        if detected:
            seen_detected.append(seen)
        for n in undetected_within:
            undetected_within[n] += not detected or seen > n

    total, accuracy = len(data["pairs"]), data["accuracy"]
    conf = {n: accuracy / (accuracy + (1 - accuracy) * undetected_within[n] / total) for n in undetected_within}
    return (statistics.mean(lengths), statistics.mean(literals), statistics.mean(seen_detected), conf[1], conf[5], conf[10])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", nargs="?", default="data_collection/data_all_in_ap_set.csv")
    parser.add_argument("--validation", action="store_true", help="Choose on a validation split of the training pairs")
    args = parser.parse_args()

    with open(args.data_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        formulas = [tuple(row) for row in reader]

    # Fixed per context and pair, so every pair of constants only recombines them
    df = build_training_data(formulas, STRATEGY, "usefulness")
    train_idx, _ = split_by_formula(df["formula_id"])
    train = df.iloc[train_idx].copy()

    context_stats = {}
    for formula_id in train["formula_id"].unique():
        by_context = {}
        for trace, _, context in generate_traces(formulas[formula_id][1], STRATEGY):
            by_context.setdefault(context, []).append(trace)
        for context, traces in by_context.items():
            context_stats[(formula_id, context)] = (trace_len(traces[0]), sum(map(count_literals, traces)) / len(traces))

    datasets = []
    if args.validation:
        inner_idx, val_idx = split_by_formula(train["formula_id"], random_state=7)
        val_ids = sorted(set(train.iloc[val_idx]["formula_id"]))
        train = train.iloc[inner_idx].copy()
        print(f"validation split: {train['formula_id'].nunique()} pairs to rank on, {len(val_ids)} to choose on")
        datasets.append(("Validation", {"pairs": [pair_data(formulas[i][0], formulas[i][1]) for i in val_ids],
                                        "accuracy": SYNTHETIC_ACCURACY}))
    datasets += [("Held-out", load_pairs("heldout_test_formulas_all_contexts_usefulness.csv", "heldout")),
                 ("Real-world", load_pairs("ALL_RL/pairs_claude.csv", "ALL_RL"))]

    header = f"{'len,lit':<12}" + "".join(f"{'| ' + name:<8}{'len':>6}{'lit':>7}{'avg':>7}{'conf1':>7}{'conf5':>7}{'conf10':>7}  " for name, _ in datasets)
    print(header)

    for len_ref in LEN_REFS:
        for lit_ref in LIT_REFS:
            train["score"] = [
                smoothed_ratio_plus(*context_stats[(f, m)], h, n, len_ref=len_ref, lit_ref=lit_ref)
                for f, m, h, n in zip(train["formula_id"], train["mutation"], train["helpful_count"], train["num_traces"])
            ]
            scores = train.groupby("mutation")["score"].mean().to_dict()
            cells = "".join(
                f"{'|':<8}{m[0]:>6.2f}{m[1]:>7.2f}{m[2]:>7.2f}{m[3]:>7.3f}{m[4]:>7.3f}{m[5]:>7.3f}  "
                for m in (measure(data, scores) for _, data in datasets)
            )
            print(f"{f'{len_ref:g},{lit_ref:g}':<12}{cells}", flush=True)


if __name__ == "__main__":
    main()
