# example-based-ltl-verifier

## Setup

Spot is not on PyPI; install it system-wide for the Python 3 you use (e.g. built from source with `./configure && make && make install`, as in the artifact's Dockerfile), then create a virtual environment that can see it:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --only-binary=:all: -r requirements.txt
.venv/bin/python -c "import spot; print(spot.version())"
```

The API keys used by `generate_gt_pairs.py` and `collect_gt_candidate.py` are read from a local `config.py` (not tracked).

## Pipeline

Spot orders formula operands by creation order within a process, so traces generated on the fly would depend on what a process handled before. The traces of every candidate are therefore generated once and read from `trace_cache.json` by all later steps:

1. `filter_synthetic_pairs.py`: drop synthetic pairs using APs outside the provided set (`data_collection/data_all_in_ap_set.csv`).
2. `precompute_traces.py`: generate the traces of all training and real-world candidates (`trace_cache.json`). By default, every mutant contributes the accepting runs of `candidate & !mutant` and `mutant & !candidate`, so all traces are accepting or rejecting with respect to the candidate; `--generator arbitrary` restores arbitrary accepting runs of the mutant and of its negation.
3. `context_rankings.py data_collection/data_all_in_ap_set.csv`: rank mutation contexts on the training pairs alone, by their detections (`freq_train_all_contexts.csv`, used by the top-k strategies) and by their mean usefulness (`context_ranking_all_contexts_smoothed.csv`, used by the `STATIC` ordering). Needs `heldout_test_formulas_all_contexts_smoothed.csv` from training `all_contexts`, to check that the split is the same.
4. `train_ordering.py data_collection/data_all_in_ap_set.csv <strategy> <smoothed|smoothed_plus>`: train the trace ordering models.
5. `evaluate.py <orderings> <pairs csv> <strategy>`: evaluate, writing `results_*` and `log_*`. In the paper, `STATIC` is LTLTrust, `STATIC_SA` is SA LTLTrust and `LTLTRUST` is Reg LTLTrust.
