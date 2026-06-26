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
python run_experiments.py --quick    # ~1–2 min smoke test (GunPoint, holdout only)
python run_experiments.py            # 15 UCR datasets, full grid (slow: many GPU cross-barcode fits)
```

Full run cost is dominated by **MTopDiv cross-barcodes**, not UCR series length: each
(query × class) pair runs 2 cross-barcodes; fit adds `n_classes × density_samples` more.
The runner prints a cross-barcode budget estimate at startup and per-method stage timings
(embedding, class clouds, self-densities, cross-persistence, RF).

Output CSV includes `time_fit_s`, `time_predict_s`, and `time_detail_json` per row.

## Input contract

- `X`: `(n_series, series_length)` univariate float array
- `y`: integer class labels
- Pass `y` to `transform` only for **training** rows (leakage exclusion by provenance index).
