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
Lite preset (`hom_dims=(0,)`, no `betti_2`): **24** per class.

## Experiments

```bash
python run_experiments.py --quick    # ~1–2 min smoke test (GunPoint, holdout only)
python run_experiments.py            # report datasets, 5 seeds, holdout + CV
python run_experiments.py --no-cv    # same grid but holdout only (skips CV, much faster)
python run_experiments.py --lite     # lite TopGen (H0-only, no betti_2) → accuracy_table_lite.csv
python run_experiments.py --seeds 0 1 2   # override default seeds (works with --quick / --no-cv)
```

### Lite TopGen preset

`--lite` sets `hom_dims=(0,)` and drops `betti_2` from `rep_names` (6 reps, H0 only).
Per-class feature count: **24** (vs 56 full). Same API; `_per_class_feature_count()` updates automatically.
Optional constructor: `TopGenTransformer.lite(...)`.

### Fusion CV (OOF CAWPE + stacking)

```bash
python run_fusion_cv.py --quick
python run_fusion_cv.py --lite
```

One train-only CV pass produces:

- `results/fusion_cv/cawpe_weights.csv` — OOF-based CAWPE weights per combo/module
- `results/fusion_cv/oof_stack/{dataset}_seed{seed}.npz` — OOF probability matrix for meta-learner
- `results/accuracy_table_fusion_cv.csv` — `CAWPE-cv:*` and `Stacking:LR:*` / `Stacking:HGB:*` rows

Weights and meta-learner are fit on train OOF only; holdout test is untouched.

### Feature independence (TopGen blocks vs baselines)

```bash
python run_independence.py --quick
python run_independence.py --full-topgen   # full TopGen instead of lite default
```

Output: `results/independence/independence.csv` — long format with `dataset, block, partner, metric, value, null_mean, p_value`.
Metrics: `residual_r2` (mean 1−R²) and `cca_mean_rho` with permutation null (≥200 by default).

### Methods are feature generators

Every method is just a **time-series feature generator**; the classifier is a separate,
shared choice (RandomForest for now — see `make_classifier`). An *experiment* is simply
which generators to concatenate before that one classifier:

| generator | features |
| --- | --- |
| `topgen`  | TopGen population-level cross-persistence (leakage-free LOO on train) |
| `catch22` | aeon catch22 |
| `tsfresh` | aeon TSFresh (the FreshPRINCE feature set, efficient profile) |

```python
EXPERIMENTS = {
    "TopGen":         ("topgen",),
    "catch22":        ("catch22",),
    "TSFresh":        ("tsfresh",),
    "TopGen+catch22": ("topgen", "catch22"),
    "TopGen+TSFresh": ("topgen", "tsfresh"),
}
```

For each dataset/seed split, the runner first computes every needed generator once
(`topgen`, `catch22`, `tsfresh`), then a second loop concatenates the requested blocks
for each experiment and fits the shared classifier. Thus `TopGen`, `TopGen+catch22`,
and `TopGen+TSFresh` reuse the same TopGen feature matrix for that split.

Full run cost is dominated by **MTopDiv cross-barcodes**, not UCR series length. TopGen's
fit runs a **leave-one-series-out (LOO)** pass: every train series is scored against every
class (2 cross-barcodes each), which simultaneously yields the frozen self-densities and the
leakage-free train features. The runner prints a cross-barcode budget estimate at startup
and per-method stage timings (embedding, class clouds, self-densities, cross-persistence, classifier).

Datasets are split into a disjoint `TUNING_DATASETS` set and a reported `REPORT_DATASETS`
set, and each result row is tagged by UCR `dataset_type` / `is_dynamical` so dynamical
(premise holds) and non-dynamical (premise does not) datasets can be reported separately.
Feature importances (permutation on the held-out split) are written to a fresh
`results/feature_importances/feature_importances.csv` — one long-format row per feature
per run, tagged with `experiment` and `source` (`topgen` / `catch22` / `tsfresh`) so you
can ask whether TopGen features keep signal next to the baselines.

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
independently. **If a cloud has fewer than `small_cloud_threshold` points (default 100),
all points are used** — no fraction downsampling. Otherwise:

- **Left / query** (one series): `clip(query_fraction × n_embedded, min_cloud_points, max_query_points)`.
- **Per-series contribution to pooled class cloud**: same rule with `per_series_fraction`.
- **Right / class** (pooled): `clip(class_fraction × |C_c|, min_cloud_points, max_class_points)`,
  per class.

Set `debug_sizes=True` on `TopGenTransformer` to print resolved cloud shapes and
cross-barcode batch sizes during fit (first LOO row) and transform (first test row).

Optional **barcode disk cache** (content-addressed on raw left/right point clouds):
set `cache_dir="/path/to/cache"` on `TopGenTransformer`. Only raw H0/H1 barcodes are
cached (`.npz` per key); `rep_names`, blocks, and Betti thresholds stay outside the
cache so ablation grids reuse the same MTopDiv results. Default `cache_dir=None` disables
caching.

The same per-class right size `M'` is used for self-density, train (LOO), and test, so sizes
stay matched (methodology §8). The right (class) cloud is deterministic per class, so
identical input series produce identical features.
