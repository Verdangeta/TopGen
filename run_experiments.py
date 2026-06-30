#!/usr/bin/env python3
"""TopGen v2 experiment runner: UCR datasets, baselines, accuracy CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zipfile
from urllib.request import urlopen

import numpy as np
from aeon.transformations.collection.feature_based import Catch22, TSFresh
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from topgen.features import ALL_REP_NAMES
from topgen.topgen import TopGenTransformer

try:
    from tqdm import tqdm
except ImportError:
    # Fallback when tqdm is not installed.
    def tqdm(iterable, **kwargs):
        return iterable


# --- experiment configuration ------------------------------------------------

# UCR "type" field per dataset. Delay embedding presumes dynamics, so the premise
# holds for dynamical types (SENSOR/MOTION/ECG/DEVICE/EOG/HEMODYNAMICS) and not
# for non-dynamical ones (SPECTRO/IMAGE). We report these groups separately.
DATASET_TYPES: dict[str, str] = {
    "GunPoint": "MOTION",
    "ItalyPowerDemand": "SENSOR",
    "ECG200": "ECG",
    "Plane": "SENSOR",
    "Lightning2": "SENSOR",
    "Earthquakes": "SENSOR",
    "Computers": "DEVICE",
    "RefrigerationDevices": "DEVICE",
    "Worms": "MOTION",
    "WormsTwoClass": "MOTION",
    "Coffee": "SPECTRO",
    "OliveOil": "SPECTRO",
    "Strawberry": "SPECTRO",
    "ArrowHead": "IMAGE",
    "Herring": "IMAGE",
}
DYNAMICAL_TYPES = frozenset({"SENSOR", "MOTION", "ECG", "DEVICE", "EOG", "HEMODYNAMICS"})


def is_dynamical(dataset: str) -> bool:
    """Whether delay embedding's dynamical premise plausibly holds for a dataset."""
    return DATASET_TYPES.get(dataset, "UNKNOWN") in DYNAMICAL_TYPES


# Disjoint dataset split for honest evaluation: tune hyperparameters ONLY on the
# tuning set (or via nested CV), then report on the held-out report set. Never
# tune on a reported dataset.
TUNING_DATASETS = (
    "ItalyPowerDemand",  # SENSOR
    "Plane",             # SENSOR
    "Worms",             # MOTION
    "Computers",         # DEVICE
    "Coffee",            # SPECTRO (non-dynamical contrast)
)
REPORT_DATASETS = (
    # Dynamical (premise holds) — primary result
    "GunPoint",
    "ECG200",
    "Lightning2",
    "Earthquakes",
    "RefrigerationDevices",
    "WormsTwoClass",
    # Non-dynamical (premise does not hold) — labelled contrast
    "OliveOil",
    "Strawberry",
    "ArrowHead",
    "Herring",
)
DATASETS = REPORT_DATASETS
SEEDS = (0, 1, 2, 3, 4)
CV_FOLDS = 5

UCR_BASE_URL = "https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
OUTPUT_CSV = "results/accuracy_table.csv"
IMPORTANCE_DIR = "results/feature_importances"
IMPORTANCE_CSV = os.path.join(IMPORTANCE_DIR, "feature_importances.csv")
IMPORTANCE_FIELDS = (
    "dataset",
    "dataset_type",
    "is_dynamical",
    "seed",
    "experiment",
    "importance_method",
    "source",
    "feature",
    "class_label",
    "rep",
    "hom_dim",
    "block",
    "importance",
    "importance_std",
)

TOPGEN_KWARGS = dict(
    rep_names=ALL_REP_NAMES,
    hom_dims=(0, 1),
    blocks=("b1", "b2", "b3"),
    class_mode="A",
    search_embedding=False,
    embedding_dimension=10,
    embedding_time_delay=4,
    per_series_fraction=0.8,
    query_fraction=0.8,
    class_fraction=0.8,
    min_cloud_points=20,
    max_query_points=500,
    max_class_points=1000,
    small_cloud_threshold=100,
    density_repeats=1,
    stride=3,
    pdist_device="cuda",
    record_timing=True,
)

# Shared classifier on top of the concatenated feature generators. The model is just
# a choice (RF for now); swap RF_KWARGS / make_classifier for e.g. RotationForest later.
RF_KWARGS = dict(n_estimators=200, n_jobs=-1)

# An experiment = which feature generators to concatenate before the shared classifier.
EXPERIMENTS = {
    "TopGen": ("topgen",),
    "catch22": ("catch22",),
    "TSFresh": ("tsfresh",),
    "TopGen+catch22": ("topgen", "catch22"),
    "TopGen+TSFresh": ("topgen", "tsfresh"),
}

QUICK_DATASETS = ("GunPoint",)
QUICK_SEEDS = (0,)
QUICK_METHODS = ("TopGen", "TopGen+catch22")
QUICK_TOPGEN_KWARGS = {
    **TOPGEN_KWARGS,
    "rep_names": ("mtd",),
}
QUICK_RF_KWARGS = dict(n_estimators=50, n_jobs=-1)


# --- why runs are slow (printed at startup) ----------------------------------


def explain_runtime_cost(
    n_train: int,
    n_test: int,
    n_classes: int,
    topgen_kwargs: dict,
    n_datasets: int,
    n_seeds: int,
    n_methods: int,
    run_cv: bool,
) -> None:
    """Print cross-barcode budget — dominant cost even on small UCR series."""
    density_repeats = topgen_kwargs.get("density_repeats", 1)
    n_reps = len(topgen_kwargs.get("rep_names", ("mtd",)))
    # LOO fit: every train series is scored vs every class (2 cross-barcodes each);
    # in-class density repeats add (R-1) more query-on-left barcodes per series.
    fit_xbarc = n_train * n_classes * 2 + n_train * max(density_repeats - 1, 0)
    test_xbarc = n_test * n_classes * 2
    per_holdout = fit_xbarc + test_xbarc
    cv_multiplier = 1 + CV_FOLDS if run_cv else 1
    grid = n_datasets * n_seeds * n_methods * cv_multiplier
    print(
        "Runtime is dominated by GPU cross-barcodes (MTopDiv), not dataset size.\n"
        f"  Per TopGen holdout (this split): ~{per_holdout} cross-barcodes "
        f"({fit_xbarc} LOO fit + {test_xbarc} test),\n"
        f"  {n_reps} linear reps each reuse the same barcode.\n"
        f"  Full grid: {n_datasets} datasets × {n_seeds} seeds × {n_methods} methods "
        f"× {cv_multiplier} (holdout{' + CV' if run_cv else ''}) = {grid} estimator fits.\n"
        "  Use --quick for a ~1–2 min smoke test."
    )


# --- UCR loading -------------------------------------------------------------


def load_ucr(
    name: str,
    data_dir: str = "data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download and load one UCR univariate dataset (train/test .txt splits)."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{name}.zip")
    if not os.path.exists(zip_path):
        with urlopen(UCR_BASE_URL.format(name=name)) as response:
            with open(zip_path, "wb") as handle:
                handle.write(response.read())

    with zipfile.ZipFile(zip_path) as archive:
        train_X, train_y = _read_ucr_split(archive, f"{name}_TRAIN.txt")
        test_X, test_y = _read_ucr_split(archive, f"{name}_TEST.txt")
    return train_X, train_y, test_X, test_y


def _read_ucr_split(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    with archive.open(member) as handle:
        rows = np.loadtxt(handle)
    labels = rows[:, 0].astype(int)
    series = rows[:, 1:]
    return series, labels


# --- sklearn adapters --------------------------------------------------------


class UnivariateToCollection(TransformerMixin, BaseEstimator):
    """Reshape (n_cases, n_timepoints) to aeon's (n_cases, 1, n_timepoints)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            return X[:, np.newaxis, :]
        return X


# --- feature generators (TS -> feature matrix) -------------------------------
#
# Every method is just a feature generator. A generator exposes:
#   fit_transform(X, y) -> train feature matrix
#   transform(X)        -> test feature matrix
#   records()           -> one descriptor per column (source + structured fields)
# Experiments concatenate generators and feed a single shared classifier.


def _clean_features(matrix):
    return np.nan_to_num(np.asarray(matrix, dtype=float), posinf=1e9, neginf=-1e9, nan=0.0)


class TopGenGenerator:
    """TopGen population-level cross-persistence features (leakage-free LOO on train)."""

    source = "topgen"

    def __init__(self, random_state, topgen_kwargs):
        self.transformer = TopGenTransformer(random_state=random_state, **topgen_kwargs)

    def fit_transform(self, X, y):
        return self.transformer.fit_transform(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

    def records(self):
        return [{**r, "source": self.source} for r in self.transformer.feature_table()]


class AeonGenerator:
    """Whole-series aeon feature transformer (catch22 / TSFresh) as a generator."""

    def __init__(self, source, transformer):
        self.source = source
        self.pipeline = Pipeline([("reshape", UnivariateToCollection()), ("feat", transformer)])
        self.n_features_ = 0

    def fit_transform(self, X, y):
        matrix = _clean_features(self.pipeline.fit_transform(X, y))
        self.n_features_ = matrix.shape[1]
        return matrix

    def transform(self, X):
        return _clean_features(self.pipeline.transform(X))

    def records(self):
        return [
            {"feature": f"{self.source}_{i}", "source": self.source,
             "class_label": "", "rep": "", "hom_dim": "", "block": ""}
            for i in range(self.n_features_)
        ]


def make_generator(name, random_state, topgen_kwargs):
    if name == "topgen":
        return TopGenGenerator(random_state, topgen_kwargs)
    if name == "catch22":
        return AeonGenerator("catch22", Catch22())
    if name == "tsfresh":
        return AeonGenerator("tsfresh", TSFresh(default_fc_parameters="efficient"))
    raise ValueError(f"Unknown feature generator: {name}")


def make_classifier(random_state, rf_kwargs):
    # The model is just a choice; swap this for RotationForest etc. later.
    return RandomForestClassifier(random_state=random_state, **rf_kwargs)


class FeatureGeneratorClassifier(BaseEstimator, ClassifierMixin):
    """Concatenate one or more TS feature generators, then a single shared classifier."""

    def __init__(self, generators=("topgen",), random_state=0, topgen_kwargs=None, rf_kwargs=None):
        self.generators = generators
        self.random_state = random_state
        self.topgen_kwargs = topgen_kwargs
        self.rf_kwargs = rf_kwargs
        self.timings_: dict[str, float] = {}

    def fit(self, X, y):
        y = np.asarray(y)
        t0 = time.perf_counter()
        topgen_kwargs = TOPGEN_KWARGS if self.topgen_kwargs is None else self.topgen_kwargs
        rf_kwargs = RF_KWARGS if self.rf_kwargs is None else self.rf_kwargs

        self.generators_ = [make_generator(n, self.random_state, topgen_kwargs) for n in self.generators]
        blocks, records = [], []
        for gen in self.generators_:
            blocks.append(gen.fit_transform(X, y))
            records += gen.records()
        self.feature_records_ = records

        self.clf_ = make_classifier(self.random_state, rf_kwargs)
        self.clf_.fit(np.hstack(blocks), y)
        self.classes_ = self.clf_.classes_
        self.timings_["fit_total"] = time.perf_counter() - t0
        self._merge_topgen_timings("fit_timings_", "topgen_fit")
        return self

    def predict(self, X):
        t0 = time.perf_counter()
        blocks = [gen.transform(X) for gen in self.generators_]
        self.last_test_features_ = np.hstack(blocks)
        y_pred = self.clf_.predict(self.last_test_features_)
        self.timings_["predict_total"] = time.perf_counter() - t0
        self._merge_topgen_timings("transform_timings_", "topgen_transform")
        return y_pred

    def _merge_topgen_timings(self, attr, prefix):
        for gen in self.generators_:
            timings = getattr(getattr(gen, "transformer", None), attr, None)
            if timings:
                for key, value in timings.items():
                    self.timings_[f"{prefix}_{key}"] = value


# --- evaluation --------------------------------------------------------------


def _method_factory(
    method_name: str,
    seed: int,
    topgen_kwargs: dict,
    rf_kwargs: dict,
):
    if method_name not in EXPERIMENTS:
        raise ValueError(f"Unknown method: {method_name}")
    return FeatureGeneratorClassifier(EXPERIMENTS[method_name], seed, topgen_kwargs, rf_kwargs)


def evaluate_holdout_timed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator,
) -> tuple[float, dict[str, float]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    timings["fit_total"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = estimator.predict(X_test)
    timings["predict_total"] = time.perf_counter() - t0

    if hasattr(estimator, "timings_"):
        timings.update(estimator.timings_)
    timings["total"] = timings["fit_total"] + timings["predict_total"]
    return float(accuracy_score(y_test, y_pred)), timings


def evaluate_cv(X: np.ndarray, y: np.ndarray, estimator, seed: int) -> float:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(np.mean(scores))


def _format_timings(timings: dict[str, float]) -> str:
    parts = []
    for key in (
        "fit_total",
        "topgen_fit_class_clouds",
        "topgen_fit_self_densities",
        "topgen_fit_cross_persistence",
        "topgen_transform_cross_persistence",
        "predict_total",
        "total",
    ):
        if key in timings:
            parts.append(f"{key}={timings[key]:.1f}s")
    return "  ".join(parts)


def run_experiments(
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    topgen_kwargs: dict,
    rf_kwargs: dict,
    run_cv: bool,
    output_csv: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    jobs = [(dataset, seed, method) for dataset in datasets for seed in seeds for method in methods]
    data_cache: dict[str, tuple[np.ndarray, ...]] = {}

    for dataset, seed, method_name in tqdm(jobs, desc="experiments", unit="job"):
        if dataset not in data_cache:
            data_cache[dataset] = load_ucr(dataset)
        train_X, train_y, test_X, test_y = data_cache[dataset]
        X_all = np.vstack([train_X, test_X])
        y_all = np.concatenate([train_y, test_y])
        n_classes = len(np.unique(train_y))

        if len(rows) == 0 or rows[-1]["dataset"] != dataset:
            print(f"\n{dataset}: train {train_X.shape}, test {test_X.shape}, classes={n_classes}")

        estimator = _method_factory(method_name, seed, topgen_kwargs, rf_kwargs)
        holdout_acc, timings = evaluate_holdout_timed(train_X, train_y, test_X, test_y, estimator)

        if (
            hasattr(estimator, "feature_records_")
            and hasattr(estimator, "clf_")
            and hasattr(estimator, "last_test_features_")
        ):
            save_feature_importances(
                estimator.clf_,
                estimator.feature_records_,
                dataset=dataset,
                seed=seed,
                dataset_type=DATASET_TYPES.get(dataset, "UNKNOWN"),
                is_dynamical=is_dynamical(dataset),
                experiment=method_name,
                out_dir=IMPORTANCE_DIR,
                X_val=estimator.last_test_features_,
                y_val=test_y,
            )

        cv_acc = float("nan")
        if run_cv:
            cv_estimator = _method_factory(method_name, seed, topgen_kwargs, rf_kwargs)
            cv_acc = evaluate_cv(X_all, y_all, cv_estimator, seed)

        row = {
            "dataset": dataset,
            "dataset_type": DATASET_TYPES.get(dataset, "UNKNOWN"),
            "is_dynamical": is_dynamical(dataset),
            "method": method_name,
            "seed": seed,
            "holdout_accuracy": holdout_acc,
            "cv_accuracy": cv_acc,
            "time_total_s": round(timings.get("total", 0.0), 2),
            "time_fit_s": round(timings.get("fit_total", 0.0), 2),
            "time_predict_s": round(timings.get("predict_total", 0.0), 2),
            "time_detail_json": json.dumps({k: round(v, 3) for k, v in sorted(timings.items())}),
        }
        rows.append(row)
        print(
            f"  {method_name:16s} seed={seed}  holdout={holdout_acc:.4f}"
            + (f"  cv={cv_acc:.4f}" if run_cv else "")
            + f"  {_format_timings(timings)}"
        )
        if output_csv is not None:
            write_csv(rows, output_csv, quiet=True)
    return rows


def save_feature_importances(
    rf,
    feature_records,
    *,
    dataset: str,
    seed: int,
    dataset_type: str,
    is_dynamical: bool,
    experiment: str,
    out_dir: str,
    importance_method: str = "permutation",
    X_val=None,
    y_val=None,
) -> None:
    if importance_method == "permutation":
        if X_val is None or y_val is None:
            raise ValueError("permutation importance requires X_val and y_val")
        result = permutation_importance(
            rf, X_val, y_val, n_repeats=10, random_state=seed, n_jobs=-1
        )
        importances = result.importances_mean
        stds = result.importances_std
    elif importance_method == "impurity":
        importances = rf.feature_importances_
        stds = np.full(len(importances), np.nan)
    else:
        raise ValueError(f"Unknown importance method: {importance_method}")

    if len(feature_records) != len(importances):
        raise ValueError(
            f"feature records {len(feature_records)} != importance length {len(importances)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(IMPORTANCE_CSV))
    write_header = not os.path.exists(path)
    rows = []
    for meta, importance, importance_std in zip(feature_records, importances, stds):
        rows.append(
            {
                "dataset": dataset,
                "dataset_type": dataset_type,
                "is_dynamical": is_dynamical,
                "seed": seed,
                "experiment": experiment,
                "importance_method": importance_method,
                "source": meta["source"],
                "feature": meta["feature"],
                "class_label": meta["class_label"],
                "rep": meta["rep"],
                "hom_dim": meta["hom_dim"],
                "block": meta["block"],
                "importance": float(importance),
                "importance_std": float(importance_std),
            }
        )
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMPORTANCE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def summarize_by_type(rows: list[dict[str, object]]) -> None:
    """Print mean holdout accuracy per method, split by dynamical vs non-dynamical."""
    print("\nMean holdout accuracy by method (dynamical premise holds vs not):")
    methods = sorted({row["method"] for row in rows})
    for group_label, predicate in (
        ("dynamical", lambda r: r["is_dynamical"]),
        ("non-dynamical", lambda r: not r["is_dynamical"]),
    ):
        print(f"  [{group_label}]")
        for method in methods:
            accs = [
                row["holdout_accuracy"]
                for row in rows
                if row["method"] == method and predicate(row)
            ]
            if accs:
                print(f"    {method:16s} {np.mean(accs):.4f}  (n={len(accs)})")


def write_csv(rows: list[dict[str, object]], path: str = OUTPUT_CSV, quiet: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_type",
        "is_dynamical",
        "method",
        "seed",
        "holdout_accuracy",
        "cv_accuracy",
        "time_total_s",
        "time_fit_s",
        "time_predict_s",
        "time_detail_json",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not quiet:
        print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TopGen v2 UCR experiment runner")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="GunPoint only, seed 0, TopGen and TopGen+catch22, holdout only, MTD reps, RF n_estimators=50",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: results/accuracy_table_quick.csv or accuracy_table.csv)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation; run the train/test holdout split only (faster).",
    )
    args = parser.parse_args()

    if args.quick:
        datasets = QUICK_DATASETS
        seeds = QUICK_SEEDS
        methods = QUICK_METHODS
        topgen_kwargs = QUICK_TOPGEN_KWARGS
        rf_kwargs = QUICK_RF_KWARGS
        run_cv = False
        output = args.output or "results/accuracy_table_quick.csv"
    else:
        datasets = DATASETS
        seeds = SEEDS
        methods = tuple(EXPERIMENTS)
        topgen_kwargs = TOPGEN_KWARGS
        rf_kwargs = RF_KWARGS
        run_cv = not args.no_cv
        output = args.output or OUTPUT_CSV

    train_X, train_y, test_X, test_y = load_ucr(datasets[0])
    explain_runtime_cost(
        n_train=train_X.shape[0],
        n_test=test_X.shape[0],
        n_classes=len(np.unique(train_y)),
        topgen_kwargs=topgen_kwargs,
        n_datasets=len(datasets),
        n_seeds=len(seeds),
        n_methods=len(methods),
        run_cv=run_cv,
    )

    rows = run_experiments(datasets, seeds, methods, topgen_kwargs, rf_kwargs, run_cv, output)
    write_csv(rows, output)
    summarize_by_type(rows)


if __name__ == "__main__":
    main()
