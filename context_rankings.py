#!/usr/bin/env python3
"""
Fixed global rankings of the mutation contexts, learned on the training pairs
alone: the same rows and the same split by formula as train_ordering.py, so
neither the real-world pairs nor the held-out synthetic pairs are involved.

  context_ranking_<strategy>_smoothed.csv: contexts by their mean usefulness,
      the regression's target without any formula features. This is what the
      STATIC ordering of evaluate.py uses (LTLTrust in the paper).
  context_ranking_<strategy>_smoothed_plus.csv: contexts by their mean
      simplicity-adjusted usefulness, for the STATIC_SA ordering (SA LTLTrust).
  freq_train_<strategy>.csv: contexts by their distinguishing traces over all
      traces of all training pairs, regardless of order. The first k of these
      are the top-k trace generation strategies, and the STATIC_DETECTIONS
      ordering uses all of them.
"""
import argparse
import csv

import pandas as pd

from train_ordering import build_training_data, split_by_formula


def write_ranking(path, header, series, pairs):
    # Ties in a fixed order, by context name
    ranked = series.reset_index()
    ranked.columns = ["context", "value"]
    ranked["pairs"] = ranked["context"].map(pairs)
    ranked = ranked.sort_values(by=["value", "context"], ascending=[False, True])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Mutation Context", header, "Training Pairs"])
        writer.writerows(ranked.itertuples(index=False, name=None))

    print(f"{path}: {len(ranked)} contexts, top: {ranked['context'].iloc[:5].tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", help="The training pairs csv")
    parser.add_argument("--strategy", default="all_contexts")
    parser.add_argument("--heldout", default="heldout_test_formulas_all_contexts_smoothed.csv",
                        help="The held-out pairs saved by train_ordering.py, to check the split against")
    args = parser.parse_args()

    with open(args.data_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        formulas = [tuple(row) for row in reader]

    df = build_training_data(formulas, args.strategy, "smoothed")
    train_idx, valid_idx = split_by_formula(df["formula_id"])

    # The same held-out pairs as the regression's
    valid_ids = sorted(set(df.iloc[valid_idx]["formula_id"]))
    heldout = pd.read_csv(args.heldout, keep_default_na=False)
    assert [(formulas[i][0], formulas[i][1]) for i in valid_ids] == list(zip(heldout["Ground Truth"], heldout["Response"]))

    train = df.iloc[train_idx]
    print(f"{train['formula_id'].nunique()} training pairs, {len(valid_ids)} held out")

    by_context = train.groupby("mutation")
    pairs = by_context["formula_id"].nunique()
    write_ranking(f"context_ranking_{args.strategy}_smoothed.csv", "Mean Usefulness", by_context["target"].mean(), pairs)
    write_ranking(f"freq_train_{args.strategy}.csv", "Detections", by_context["helpful_count"].sum(), pairs)

    # The same pairs, scored by their simplicity-adjusted usefulness
    df_sa = build_training_data(formulas, args.strategy, "smoothed_plus")
    assert df_sa[["formula_id", "mutation"]].equals(df[["formula_id", "mutation"]])
    train_sa = df_sa.iloc[train_idx].groupby("mutation")
    write_ranking(f"context_ranking_{args.strategy}_smoothed_plus.csv", "Mean SA Usefulness", train_sa["target"].mean(), pairs)

    detections = by_context["helpful_count"].sum()
    print(f"{len(detections)} contexts on the training pairs, {(detections > 0).sum()} distinguished at least one pair")


if __name__ == "__main__":
    main()
