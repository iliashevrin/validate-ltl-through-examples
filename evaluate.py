#!/usr/bin/env python3
import sys
import spot
spot.setup()
import itertools
from mutation_based import generate_traces
from utils import check_acceptance, get_formula_features, collect_aps, trace_len, count_literals
from train_ordering import trace_ranking

import csv
import random
import statistics
import joblib
import numpy as np



DATASIZE = {
    
    # Total candidates per dataset, as reported by count_excluded.py
    "textbook":367,
    "ARTEMIS":122,
    "SpaceWire":36,
    "Dwyer":99,

    "ALL_RL":624,

    "ConformalLTL":678,
    "Synthetic":1000,

    "heldout": 2542 # Approximation: 20% of the 12711 synthetic candidates using only provided APs
}




def evaluate(cand, gt, traces):

    log = ""

    seen = 0
    length = 0
    lit = 0
    acc_values = {"True": 0, "False": 0, "None": 0}

    log += f"Candidate = {cand}\n"
    log += f"GT = {gt}\n"

    good_mc = None

    # Translated once per pair, large ground truths take seconds to translate
    gt_aut = spot.translate(gt)

    for index, t in enumerate(traces):

        (trace, cand_acc, mc) = t

        # if cand == "G(x1 -> X(x1))":
        #     print(t)

        seen += 1
        length += trace_len(trace)
        lit += count_literals(trace)

        acc = check_acceptance(gt_aut, trace)

        acc_values[str(acc)] += 1

        log += f"trace#{index}, cand_acc={cand_acc}, gt_acc={acc}, {trace}\n"

        if acc != cand_acc:
            good_mc = mc
            break

    log += "\n"

    avg_len = length/seen if seen > 0 else 0
    avg_lit = lit/seen if lit > 0 else 0

    return seen, avg_len, avg_lit, good_mc, acc_values, log




def stats(data):

    if len(data) <= 1:
        return f"N/A"

    avg_val = statistics.mean(data)      # Average (Arithmetic Mean)
    med_val = statistics.median(data)    # Median
    std_val = statistics.stdev(data)     # Standard Deviation (sample)
    max_val = max(data)                  # Built-in max function
    min_val = min(data)                  # Built-in min function

    # Percentiles
    p75 = np.percentile(data, 75)
    p90 = np.percentile(data, 90)

    return (
        f"Avg: {avg_val:.3f}, "
        f"Median: {med_val:.2f}, "
        f"Std: {std_val:.2f}, "
        f"P75: {p75:.2f}, "
        f"P90: {p90:.2f}, "
        f"Max: {max_val:.2f}, "
        f"Min: {min_val:.2f}"
    )



def confidence(undetected_under_n, total, dataset):

    mismatch = total / DATASIZE[dataset]
    match = (1 - mismatch)

    return match / (match + (mismatch * (undetected_under_n / total)))



class OrderingData:

    def __init__(self, name, order, repeats):

        self.name = name
        self.order = order
        self.repeats = repeats

        self.seen_dt = []
        self.seen_ndt = []
        self.lengths = []
        self.lits = []
        self.total = 0
        self.mc_map = {}
        self.acc_values = {"True": 0, "False": 0, "None": 0}
        self.full_log = ""




def by_length(formula, traces, strategy):
    traces.sort(key=lambda trace: trace_len(trace[0]))
    return traces

def ordering_random(formula, traces, strategy):
    random.shuffle(traces)
    return traces

# Optional suffix of the model files (see train_ordering.py --tag), set from argv[4]
MODEL_TAG = ""

def ltltrust(formula, traces, strategy):
    return trace_ranking(formula, traces, strategy, "smoothed" + MODEL_TAG)

def ltltrust_plus(formula, traces, strategy):
    return trace_ranking(formula, traces, strategy, "smoothed_plus" + MODEL_TAG)



def main():

    global MODEL_TAG
    MODEL_TAG = sys.argv[4] if len(sys.argv) > 4 else ""

    ordering_names = sys.argv[1].split(",")
    orderings = []

    for name in ordering_names:

        if name == "RANDOM":
            orderings.append(OrderingData(name, ordering_random, 30))
        elif name == "BY_LENGTH":
            orderings.append(OrderingData(name, by_length, 1))
        elif name == "LTLTRUST":
            orderings.append(OrderingData(name, ltltrust, 1))
        elif name == "LTLTRUST_PLUS":
            orderings.append(OrderingData(name, ltltrust_plus, 1))
        else:
            raise ValueError("Incorrect ordering function")

    dataset = None
    for key in DATASIZE.keys():
        if key in sys.argv[2]:
            dataset = key

    if dataset is None:
        raise ValueError("Incorrect dataset")

    strategy = sys.argv[3]

    total = 0

    # mp_map = {}

    with open(sys.argv[2], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt = row["Ground Truth"]
            cand = row["Response"]

            # key = f"GT={spot.mp_class(gt, 'v')}; Cand={spot.mp_class(cand, 'v')}"
            # print(key)
            # if key not in mp_map:
            #     mp_map[key] = []

            total += 1

            print(f"gt={gt}; cand={cand}")

            traces = generate_traces(cand, strategy)

            if not traces:
                for data in orderings:
                    data.seen_ndt.append(0)
                    data.lengths.append(0)

                continue

            for data in orderings:

                print(f"ordering={data.name}")

                curr_seen_dt = []
                curr_length = []
                curr_lit = []

                for _ in range(0, data.repeats):

                    traces = data.order(cand, traces, strategy)

                    seen, length, lit, mc, values, log = evaluate(cand, gt, traces)

                    # mp_map[key].append(seen)

                    for val in values:
                        data.acc_values[val] += values[val]

                    curr_length.append(length)
                    curr_lit.append(lit)

                    data.full_log += log

                    # Mismatch was successfully detected, look at the mc of the discriminating trace
                    if mc is not None:
                        curr_seen_dt.append(seen)

                        if mc not in data.mc_map:
                            data.mc_map[mc] = 1
                        else:
                            data.mc_map[mc] += 1

                    else:
                        # Undetected case does not depend on order, so no need to run it 30 times
                        data.seen_ndt.append(seen)
                        break


                if len(curr_seen_dt) > 0:
                    data.seen_dt.append(statistics.mean(curr_seen_dt))

                data.lengths.append(statistics.mean(curr_length))
                data.lits.append(statistics.mean(curr_lit))



    accuracy = 1 - (total / DATASIZE[dataset])

    for data in orderings:

        log_file = f"log_{dataset}_{data.name}_{strategy}{MODEL_TAG}.txt"
        with open(log_file, "w", encoding="utf-8") as out:
            out.write(data.full_log)


        det_more_than_i = []
        undet_under_i = []
        confidence = []    
        undet_inf = len(data.seen_ndt) / total

        output_file = f"results_{dataset}_{data.name}_{strategy}{MODEL_TAG}.txt"

        with open(output_file, "w", encoding="utf-8") as out:
            print(f'Accepting traces for GT: {(data.acc_values["True"] / data.repeats):.3f}', file=out)
            print(f'Rejecting traces for GT: {(data.acc_values["False"] / data.repeats):.3f}', file=out)
            print(f'Inconclusive traces for GT: {(data.acc_values["None"] / data.repeats):.3f}', file=out)

            print(f'\nAccuracy: {accuracy:.3f}', file=out)
            print(f'Detected Ratio: {(1-undet_inf):.3f}', file=out)

            for i in range(0, 16):
                det_more_than_i.append(len([n for n in data.seen_dt if n > i]) / total)
                undet_under_i.append(undet_inf + det_more_than_i[i])
                confidence.append(
                    accuracy / (accuracy + ((1 - accuracy) * undet_under_i[i]))
                )

                print(
                    f'\nUndetected <={i} Ratio: {undet_under_i[i]:.3f} '
                    f'(Undetected Ratio: {undet_inf:.3f} + '
                    f'Detected >{i} Ratio: {det_more_than_i[i]:.3f})',
                    file=out,
                )
                print(
                    f'Confidence score for n={i}: {confidence[i]:.3f}',
                    file=out,
                )

            print(f'\nTraces to Detection: {stats(data.seen_dt)}', file=out)
            # print(f'Inspected Traces When No Detection: {stats(data.seen_ndt)}', file=out)

            print(f'\nTrace Lengths: {stats(data.lengths)}', file=out)
            print(f'\nTrace Literals: {stats(data.lits)}', file=out)

            print(f'\nMutation Contexts by Usefulness', file=out)

            mc_map = dict(sorted(data.mc_map.items(), key=lambda item: item[1]))
            for mc in mc_map:
                print(f'{mc}:{mc_map[mc]}', file=out)

            print(f'\nTraces to Detection List: {[f"{seen:.2f}" for seen in data.seen_dt]}', file=out)



if __name__ == "__main__":
    main()