#!/usr/bin/env python3
"""
The orderings under an engineer who is not a perfect oracle, with no retraining.
The engineer stops at the first trace they flag as a mismatch. A trace that truly
distinguishes the pair is flagged with probability hit, and any other trace with
probability false_alarm.

  label: the acceptance label is wrong with probability e, and is then one of the
         two other labels at random. A distinguishing trace goes unflagged only if
         the wrong label happens to be the candidate's, so hit = 1 - e/2.
  flip:  the verdict is wrong with probability e, so hit = 1 - e.

In both, false_alarm = e. Every number is exact, including over the random orders,
and --trials cross-checks them by simulation. With e = 0 this reproduces the number
of traces to detection and the detection ratio of evaluate.py.
"""
import argparse
import csv

import numpy as np
from scipy.stats import wilcoxon

from evaluate import by_length, ltltrust, static_sa_usefulness, static_usefulness
from mutation_based import generate_traces
from utils import check_acceptance, translate


FIXED_ORDERINGS = [
    ("Shortest First", by_length),
    ("LTLTrust", static_usefulness),
    ("SA LTLTrust", static_sa_usefulness),
    ("Reg LTLTrust", ltltrust),
]
ORDERINGS = ["Random"] + [name for name, _ in FIXED_ORDERINGS]


def collect(pairs_path, strategy):
    """Per pair and ordering, whether each trace distinguishes the pair, in the order shown."""
    pairs = []
    with open(pairs_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            traces = generate_traces(row["Response"], strategy)
            gt = translate(row["Ground Truth"])
            distinguishes = {(t, acc): check_acceptance(gt, t) != acc for t, acc, _ in traces}
            entry = {"counts": (len(traces), sum(distinguishes.values()))}
            for name, order in FIXED_ORDERINGS:
                ordered = order(row["Response"], list(traces), strategy) if traces else []
                entry[name] = [int(distinguishes[(t, acc)]) for t, acc, _ in ordered]
            pairs.append(entry)
    return pairs


def fixed_order(flags, hit, false_alarm):
    """(P[correct mismatch], P[false mismatch], E[position; a mismatch is flagged])"""
    survive, correct, wrong, position = 1.0, 0.0, 0.0, 0.0
    for i, distinguishes in enumerate(flags, start=1):
        rate = hit if distinguishes else false_alarm
        if distinguishes:
            correct += survive * rate
        else:
            wrong += survive * rate
        position += i * survive * rate
        survive *= 1 - rate
    return correct, wrong, position


def random_order_tables(max_u, max_v, hit, false_alarm):
    """The same over a uniformly random order, with u other and v distinguishing traces left."""
    correct = np.zeros((max_u + 1, max_v + 1))
    wrong = np.zeros((max_u + 1, max_v + 1))
    position = np.zeros((max_u + 1, max_v + 1))
    for u in range(max_u + 1):
        for v in range(max_v + 1):
            if u + v == 0:
                continue
            p_d, p_o = v / (u + v), u / (u + v)
            c, w, g = 0.0, 0.0, 0.0
            if v:
                c += p_d * (hit + (1 - hit) * correct[u, v - 1])
                w += p_d * (1 - hit) * wrong[u, v - 1]
                g += p_d * (hit + (1 - hit) * (position[u, v - 1] + correct[u, v - 1] + wrong[u, v - 1]))
            if u:
                c += p_o * (1 - false_alarm) * correct[u - 1, v]
                w += p_o * (false_alarm + (1 - false_alarm) * wrong[u - 1, v])
                g += p_o * (false_alarm + (1 - false_alarm) * (position[u - 1, v] + correct[u - 1, v] + wrong[u - 1, v]))
            correct[u, v], wrong[u, v], position[u, v] = c, w, g
    return correct, wrong, position


def exact(pairs, hit, false_alarm):
    """{ordering: per-pair array of (correct, wrong, position)}"""
    max_u = max(p["counts"][0] - p["counts"][1] for p in pairs)
    max_v = max(p["counts"][1] for p in pairs)
    tables = random_order_tables(max_u, max_v, hit, false_alarm)
    out = {"Random": np.array([[t[n - d, d] for t in tables] for n, d in (p["counts"] for p in pairs)])}
    for name, _ in FIXED_ORDERINGS:
        out[name] = np.array([fixed_order(p[name], hit, false_alarm) for p in pairs])
    return out


def simulate(pairs, hit, false_alarm, trials, seed=0):
    """{ordering: (P[correct], P[false], mean position when flagged)} over simulated trials."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ORDERINGS:
        correct = wrong = positions = 0
        for p in pairs:
            n, d = p["counts"]
            if n == 0:
                continue
            for _ in range(trials):
                if name == "Random":
                    flags = np.zeros(n, dtype=bool)
                    flags[rng.choice(n, d, replace=False)] = True
                else:
                    flags = np.array(p[name], dtype=bool)
                hits = np.flatnonzero(rng.random(n) < np.where(flags, hit, false_alarm))
                if hits.size:
                    positions += hits[0] + 1
                    if flags[hits[0]]:
                        correct += 1
                    else:
                        wrong += 1
        total = trials * len(pairs)
        out[name] = (correct / total, wrong / total, positions / (correct + wrong) if correct + wrong else float("nan"))
    return out


def table(pairs, result):
    print(f"{'ordering':<26}{'traces to any mismatch':>24}{'P[correct mismatch]':>21}{'P[false mismatch]':>19}{'P[none flagged]':>17}{'P[correct | flagged]':>22}")
    for name in ORDERINGS:
        c, w, g = result[name].sum(axis=0) / len(pairs)
        print(f"{name:<26}{g / (c + w):>24.3f}{c:>21.3f}{w:>19.3f}{1 - c - w:>17.3f}{c / (c + w):>22.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="?", default="ALL_RL/pairs_claude.csv")
    parser.add_argument("--strategy", default="all_contexts")
    parser.add_argument("--error", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=0, help="Simulated trials per pair, to cross-check the exact numbers")
    args = parser.parse_args()

    pairs = collect(args.pairs, args.strategy)
    e = args.error
    print(f"{len(pairs)} pairs, {sum(p['counts'][1] > 0 for p in pairs)} detectable")

    print("\n=== Perfect oracle")
    table(pairs, exact(pairs, 1.0, 0.0))

    for label, hit in [("wrong label", 1 - e / 2), ("flipped verdict", 1 - e)]:
        print(f"\n=== e = {e}, {label} (hit {hit}, false alarm {e})")
        result = exact(pairs, hit, e)
        table(pairs, result)

        if args.trials:
            print(f"--- simulated, {args.trials} trials per pair")
            for name, (c, w, g) in simulate(pairs, hit, e, args.trials).items():
                print(f"{name:<26}{g:>24.3f}{c:>21.3f}{w:>19.3f}")

        print("--- paired Wilcoxon on the per-pair P[correct mismatch], against LTLTrust")
        base = result["LTLTrust"][:, 0]
        for name in ORDERINGS:
            if name != "LTLTrust":
                other = result[name][:, 0]
                print(f"{name:<26} mean diff {np.mean(other - base):+.3f}  p={wilcoxon(other, base).pvalue:.2e}")


if __name__ == "__main__":
    main()
