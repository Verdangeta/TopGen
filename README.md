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
python run_experiments.py            # report datasets, 5 seeds (slow: many GPU cross-barcode fits)
```

Full run cost is dominated by **MTopDiv cross-barcodes**, not UCR series length. Fit runs
a **leave-one-series-out (LOO)** pass: every train series is scored against every class
(2 cross-barcodes each), which simultaneously yields the frozen self-densities and the
leakage-free train features. The runner prints a cross-barcode budget estimate at startup
and per-method stage timings (embedding, class clouds, self-densities, cross-persistence, RF).

Datasets are split into a disjoint `TUNING_DATASETS` set and a reported `REPORT_DATASETS`
set, and each result row is tagged by UCR `dataset_type` / `is_dynamical` so dynamical
(premise holds) and non-dynamical (premise does not) datasets can be reported separately.
RandomForest feature importances are saved per run under `results/feature_importances/`.

Output CSV includes `dataset_type`, `is_dynamical`, `time_fit_s`, `time_predict_s`, and
`time_detail_json` per row.

## Input contract

- `X`: `(n_series, series_length)` univariate float array
- `y`: integer class labels
- `fit(X, y)` runs the LOO pass; `fit_transform(X, y)` returns the cached leakage-free
  train features. `transform(X)` is **out-of-sample only** (single-series query vs frozen,
  deterministic class clouds) — do not pass train rows back through `transform`.

## Cloud sizes (asymmetric, fraction-based)

Cross-barcodes are asymmetric, so the left (query) and right (class) clouds are sized
independently:

- **Left / query** = `clip(query_fraction * per-series embedded points, min_cloud_points, max_query_points)`.
- **Right / class** = `clip(class_fraction * |pooled class cloud|, min_cloud_points, max_class_points)`,
  computed per class (the class cloud can be far larger than one series, hence its own
  higher cap).

The same per-class right size `M'` is used for self-density, train (LOO), and test, so sizes
stay matched (methodology §8). The right (class) cloud is deterministic per class, so
identical input series produce identical features.
