from mutation_based import generate_traces
from utils import simulate_user
import sys
import csv
import json
from evaluate import DATASIZE


def collect_frequencies(
    formulas,
    strategy,
):
    """
    Count, per mutation context, the generated traces that distinguish the
    candidate from the ground truth, over all traces of all pairs, i.e.,
    regardless of the order in which the traces would be examined. Contexts
    that never distinguish a pair are kept with a count of zero.
    """
    by_mutation = {}

    for formula_id, formula in enumerate(formulas):

        # formula = (ground truth, candidate)

        # trace = (trace, accept/reject, mutation type)

        traces = generate_traces(formula[1], strategy)

        for trace in traces:
            if trace[2] not in by_mutation:
                by_mutation[trace[2]] = 0

            if simulate_user(formula[0], trace[0], trace[1]):
                by_mutation[trace[2]] += 1

    return by_mutation


dataset = None
for key in DATASIZE.keys():
    if key in sys.argv[1]:
        dataset = key

if dataset is None:
    raise ValueError("Incorrect dataset")


with open(sys.argv[1], "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    formulas = [tuple(row) for row in reader]

freq = collect_frequencies(formulas, sys.argv[2])

freq = dict(sorted(freq.items(), key=lambda item: -item[1]))


with open(f"freq_{dataset}_{sys.argv[2]}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Mutation Context", "Detections"])  # Optional: write a header

    for key, value in freq.items():
        writer.writerow([key, value])
