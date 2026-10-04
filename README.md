# example-based-ltl-verifier

## Setup

Spot is not on PyPI; install it system-wide for the Python 3 you use (e.g. built from source with `./configure && make && make install`, as in the artifact's Dockerfile), then create a virtual environment that can see it:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --only-binary=:all: -r requirements.txt
.venv/bin/python -c "import spot; print(spot.version())"
```

The API keys used by `generate_gt_pairs.py` and `collect_gt_candidate.py` are read from a local `config.py` (not tracked).
