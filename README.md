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
2. `precompute_traces.py`: generate the traces of all training and real-world candidates (`trace_cache.json`).
3. `collect_frequencies.py ALL_RL/pairs_claude.csv all_contexts`: rank mutation contexts by their detections on the real-world pairs (`freq_ALL_RL_all_contexts.csv`, used by the top-k strategies).
4. `train_ordering.py data_collection/data_all_in_ap_set.csv <strategy> <smoothed|smoothed_plus>`: train the trace ordering models.
5. `evaluate.py <orderings> <pairs csv> <strategy>`: evaluate, writing `results_*` and `log_*`.
