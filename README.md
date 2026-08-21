# TopGen-IC: Inter-Class Cross-Persistent Homology Features for Time-Series Classification

Population-level features for time-series classification.

TopGen turns each time series into a cloud of short, overlapping states and compares it
with the typical cloud of every class. Cross-persistence summarizes these comparisons as
ordinary numeric features, which can then be passed to any tabular classifier.

`time series → point cloud → class comparisons → feature vector → classifier`

## Requirements

- Python 3
- NVIDIA GPU with CUDA (`MTopDiv`/`ripserplusplus` is GPU-only)
- Internet access to download the example UCR datasets

## Setup

```bash
git clone https://github.com/IlyaTrofimov/MTopDiv external/MTopDiv
python -m pip install -e external/MTopDiv
python -m pip install -r requirements.txt
```

If `giotto-tda` conflicts with a newer `scikit-learn`, install it without dependency
resolution:

```bash
python -m pip install numpy scipy "scikit-learn>=1.6" matplotlib aeon tsfresh tqdm
python -m pip install --no-deps giotto-tda
python -m pip install plotly python-igraph pyflagser
```

## Quick start

```bash
python run_minimal_demo.py
```

The demo downloads GunPoint, fits TopGen, and prints the train/test feature matrices.

## Experiments

```bash
python run_experiments.py --quick       # one-dataset smoke test
python run_experiments.py               # full holdout + cross-validation run
python run_experiments.py --no-cv       # faster: holdout only
python run_experiments.py --lite        # H0-only TopGen
python run_experiments.py --seeds 0 1 2
```

Additional analyses:

```bash
python run_fusion_cv.py --quick          # CAWPE and stacking
python run_independence.py --quick       # TopGen blocks vs baselines
```

Results are written to `results/`. The experiment runner compares TopGen with `catch22`
and `TSFresh`, and also tests their combinations using the same classifier.

## Python API

```python
from topgen import TopGenTransformer

model = TopGenTransformer.lite(pdist_device="cuda")
X_train_features = model.fit_transform(X_train, y_train)
X_test_features = model.transform(X_test)
```

Input arrays have shape `(n_series, series_length)`; labels are a one-dimensional array.
Use `fit_transform` for training rows and `transform` only for unseen rows. During training,
TopGen leaves the current series out of its class cloud to prevent target leakage.

The three feature blocks have a simple interpretation:

- `B1` — comparison scores in both directions;
- `B2` — asymmetry between those directions;
- `B3` — how typical the score is for a class.

## Layout

```text
topgen/
  vectorizers.py   # scalar summaries
  clouds.py        # embeddings and class clouds
  features.py      # cross-persistence feature blocks
  topgen.py        # sklearn-compatible transformer
run_minimal_demo.py
run_experiments.py
```

Cross-barcode computation dominates runtime. Set `cache_dir` on `TopGenTransformer` to
reuse previously computed barcodes.
