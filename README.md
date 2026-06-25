# TopGen v2

Population-level topological features for time series classification.

## Layout

```
topgen/
  vectorizers.py   # level-1 vectorizers (persistence stats, log-signature, catch22)
  clouds.py        # giotto-tda embedding, subsampling, provenance-tracked class clouds
  features.py      # MTopDiv cross-barcode wrapper, 3 feature blocks, self-densities
  topgen.py        # TopGenTransformer (sklearn API)
run_minimal_demo.py
run_experiments.py
```

## Setup

```bash
git clone https://github.com/IlyaTrofimov/MTopDiv external/MTopDiv
pip install -e external/MTopDiv
pip install -r requirements.txt
```

**CUDA is required** for cross-barcode computation via MTopDiv/ripserplusplus.
Use `pdist_device="cuda"` (default) or `pdist_device="cuda:0"`.

Takens embedding uses **giotto-tda** (`SingleTakensEmbedding` + `TakensEmbedding`),
matching `Topological_classifier.py` from the TDA_experiments repo.

## Minimal demo (R={MTD}, Mode A, H0+H1)

```bash
python run_minimal_demo.py
```

This fits `TopGenTransformer` on GunPoint and prints train/test feature matrices.
Per-class feature count with `R={MTD}`: **8** (4 B1 + 2 B2 + 2 B3) × number of classes.
With all seven representations (`ALL_REP_NAMES`): **56** per class.

## Experiments

```bash
python run_experiments.py
```

Runs TopGen (full 7 representations), catch22, FreshPRINCE, and TopGen+catch22 on
GunPoint, Coffee, and ItalyPowerDemand. Writes `results/accuracy_table.csv`.

## Input contract

- `X`: `(n_series, series_length)` univariate float array
- `y`: integer class labels
- Pass `y` to `transform` only for **training** rows (leakage exclusion by provenance index).
