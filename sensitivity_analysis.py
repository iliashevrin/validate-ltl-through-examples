#!/usr/bin/env python3
"""
Sensitivity of the confidence score to misestimating the NL2LTL accuracy.

conf_n = Acc / (Acc + (1 - Acc) * NDT_n), where NDT_n is the proportion of
Cand-GT pairs not detected within n traces. NDT_n is measured by evaluate.py
and does not depend on Acc, so conf_n can be recomputed for an assumed
accuracy Acc +- delta while keeping NDT_n fixed.
"""
import argparse
import csv
import re

# Total candidates per dataset, as in evaluate.py's DATASIZE
DATASETS = {
    "Dwyer": ("Dwyer/pairs_claude.csv", "results_Dwyer_LTLTRUST_all_contexts.txt", 99),
    "SpaceWire": ("SpaceWire/pairs_claude.csv", "results_SpaceWire_LTLTRUST_all_contexts.txt", 36),
    "Textbook": ("textbook/pairs_claude.csv", "results_textbook_LTLTRUST_all_contexts.txt", 367),
    "ARTEMIS": ("ARTEMIS/pairs_claude.csv", "results_ARTEMIS_LTLTRUST_all_contexts.txt", 122),
}


def confidence(acc, ndt):
    return acc / (acc + (1 - acc) * ndt)


def measure(pairs_file, results_file, candidates, n):
    """Exact accuracy and NDT_n of a dataset, from its pairs and evaluate.py results."""
    with open(pairs_file, newline="", encoding="utf-8") as f:
        mismatches = sum(1 for _ in csv.DictReader(f))

    with open(results_file, encoding="utf-8") as f:
        txt = f.read()

    detected = [float(x) for x in re.findall(r"'([\d.]+)'", re.search(r"Traces to Detection List: \[(.*)\]", txt).group(1))]
    printed_conf = float(re.search(rf"Confidence score for n={n}: ([\d.]+)", txt).group(1))

    undetected_within_n = (mismatches - len(detected)) + sum(1 for seen in detected if seen > n)

    return 1 - mismatches / candidates, undetected_within_n / mismatches, printed_conf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=5, help="Number of examined traces")
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2],
                        help="Absolute accuracy misestimations to evaluate")

    args = parser.parse_args()

    print(f"conf_{args.n} under an assumed accuracy Acc - delta .. Acc + delta (LTLTrust ordering)\n")

    header = f"{'Dataset':<11}{'Acc':>7}{'NDT':>7}{'conf':>7}" + "".join(f"{f'+-{d:g}':>17}" for d in args.deltas)
    print(header)

    for name, (pairs_file, results_file, candidates) in DATASETS.items():
        acc, ndt, printed_conf = measure(pairs_file, results_file, candidates, args.n)
        conf = confidence(acc, ndt)

        assert abs(conf - printed_conf) < 5e-4, f"{name}: computed {conf:.4f}, evaluate.py printed {printed_conf}"

        cells = []
        for delta in args.deltas:
            low, high = acc - delta, acc + delta
            if low <= 0 or high >= 1:
                cells.append(f"{'n/a':>17}")
            else:
                cells.append(f"{confidence(low, ndt):>8.3f}-{confidence(high, ndt):.3f}")

        print(f"{name:<11}{acc:>7.3f}{ndt:>7.3f}{conf:>7.3f}" + "".join(cells))

    print("\nRanges are conf at Acc - delta (underestimated accuracy) to conf at Acc + delta (overestimated accuracy);")
    print("n/a: the interval leaves (0, 1).")


if __name__ == "__main__":
    main()
