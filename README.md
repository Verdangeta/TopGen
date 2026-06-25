# TopGen v2

Population-level topological features for time series classification.

## Layout

```
topgen/
  vectorizers.py   # level-1 vectorizers (persistence stats, log-signature, catch22)
  clouds.py        # delay embedding, subsampling, provenance-tracked class clouds
  features.py      # MTopDiv cross-barcode wrapper, 3 feature blocks, self-densities
  topgen.py        # TopGenTransformer (sklearn API)
run_minimal_demo.py
```

## Setup

```bash
git clone https://github.com/IlyaTrofimov/MTopDiv external/MTopDiv
pip install -e external/MTopDiv
pip install -r requirements.txt
```

MTopDiv requires **CUDA + ripserplusplus** at runtime (see `external/MTopDiv/README.md`).
On CPU-only hosts the library probes MTopDiv once and falls back to a `ripser`-based
path with the same augmented-distance construction when the probe fails.

## Minimal demo (R={MTD}, Mode A, H0+H1)

```bash
python run_minimal_demo.py
```

This fits `TopGenTransformer` on GunPoint and prints train/test feature matrices.
Per-class feature count with the minimal config: **8** (4 B1 + 2 B2 + 2 B3) × number of classes.

## Input contract

- `X`: `(n_series, series_length)` univariate float array
- `y`: integer class labels
- Pass `y` to `transform` only for **training** rows (leakage exclusion by provenance index).
