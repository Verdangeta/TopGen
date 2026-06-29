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
from aeon.classification.feature_based import FreshPRINCEClassifier
from aeon.transformations.collection.feature_based import Catch22
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

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

TOPGEN_KWARGS = dict(
    rep_names=ALL_REP_NAMES,
    hom_dims=(0, 1),
    blocks=("b1", "b2", "b3"),
    class_mode="A",
    search_embedding=False,
    embedding_dimension=10,
    embedding_time_delay=4,
    per_series_fraction=1.0,
    query_fraction=0.6,
    class_fraction=0.8,
    min_cloud_points=20,
    max_query_points=500,
    max_class_points=1000,
    density_repeats=1,
    stride=3,
    pdist_device="cuda",
    record_timing=True,
)

RF_KWARGS = dict(n_estimators=200, n_jobs=-1)

# Independent baseline: whole-series TSFresh → RotationForest (not used inside TopGen clouds).
FRESHPRINCE_KWARGS = dict(default_fc_parameters="efficient", verbose=0)

QUICK_DATASETS = ("GunPoint",)
QUICK_SEEDS = (0,)
QUICK_METHODS = ("TopGen", "catch22")
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


class AeonCatch22Pipeline(BaseEstimator, ClassifierMixin):
    """catch22 features + random forest baseline."""

    def __init__(self, random_state: int = 0, rf_kwargs: dict | None = None):
        self.random_state = random_state
        self.rf_kwargs = RF_KWARGS if rf_kwargs is None else rf_kwargs

    def fit(self, X, y):
        self.pipeline_ = Pipeline(
            [
                ("reshape", UnivariateToCollection()),
                ("catch22", Catch22()),
                ("rf", RandomForestClassifier(random_state=self.random_state, **self.rf_kwargs)),
            ]
        )
        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.named_steps["rf"].classes_
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)


class FreshPRINCEBaseline(BaseEstimator, ClassifierMixin):
    """Whole-series TSFresh features + RotationForest (aeon), independent of TopGen."""

    def __init__(
        self,
        random_state: int = 0,
        rf_kwargs: dict | None = None,
        freshprince_kwargs: dict | None = None,
    ):
        self.random_state = random_state
        self.rf_kwargs = RF_KWARGS if rf_kwargs is None else rf_kwargs
        self.freshprince_kwargs = FRESHPRINCE_KWARGS if freshprince_kwargs is None else freshprince_kwargs

    def fit(self, X, y):
        self.estimator_ = FreshPRINCEClassifier(
            n_estimators=self.rf_kwargs["n_estimators"],
            random_state=self.random_state,
            n_jobs=self.rf_kwargs["n_jobs"],
            **self.freshprince_kwargs,
        )
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        return self.estimator_.predict(X)


class TopGenRFClassifier(BaseEstimator, ClassifierMixin):
    """TopGen features + random forest with staged timing."""

    def __init__(
        self,
        random_state: int = 0,
        topgen_kwargs: dict | None = None,
        rf_kwargs: dict | None = None,
    ):
        self.random_state = random_state
        self.topgen_kwargs = TOPGEN_KWARGS if topgen_kwargs is None else topgen_kwargs
        self.rf_kwargs = RF_KWARGS if rf_kwargs is None else rf_kwargs
        self.timings_: dict[str, float] = {}

    def fit(self, X, y):
        y = np.asarray(y)
        total_start = time.perf_counter()

        self.topgen_ = TopGenTransformer(random_state=self.random_state, **self.topgen_kwargs)
        # Single LOO pass: fit_transform returns the cached leakage-free train matrix.
        t0 = time.perf_counter()
        X_train = self.topgen_.fit_transform(X, y)
        self.timings_["topgen_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.rf_ = RandomForestClassifier(random_state=self.random_state, **self.rf_kwargs)
        self.rf_.fit(X_train, y)
        self.timings_["rf_fit"] = time.perf_counter() - t0

        # Feature importances aligned with TopGen feature names (Fix 5: confirm
        # TopGen features are used, not overfit).
        self.feature_importances_ = self.rf_.feature_importances_
        self.feature_names_ = list(self.topgen_.get_feature_names_out())

        if hasattr(self.topgen_, "fit_timings_"):
            for key, value in self.topgen_.fit_timings_.items():
                self.timings_[f"topgen_fit_{key}"] = value

        self.timings_["fit_total"] = time.perf_counter() - total_start
        self.classes_ = self.rf_.classes_
        return self

    def predict(self, X):
        predict_start = time.perf_counter()
        t0 = time.perf_counter()
        X_test = self.topgen_.transform(X)
        self.timings_["topgen_transform_test"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = self.rf_.predict(X_test)
        self.timings_["rf_predict"] = time.perf_counter() - t0

        if hasattr(self.topgen_, "transform_timings_"):
            for key, value in self.topgen_.transform_timings_.items():
                self.timings_[f"topgen_transform_test_{key}"] = value

        self.timings_["predict_total"] = time.perf_counter() - predict_start
        return y_pred


class TopGenAddonClassifier(BaseEstimator, ClassifierMixin):
    """TopGen + catch22 via FeatureUnion, then random forest."""

    def __init__(self, random_state: int = 0, topgen_kwargs: dict | None = None, rf_kwargs: dict | None = None):
        self.random_state = random_state
        self.topgen_kwargs = TOPGEN_KWARGS if topgen_kwargs is None else topgen_kwargs
        self.rf_kwargs = RF_KWARGS if rf_kwargs is None else rf_kwargs
        self.timings_: dict[str, float] = {}

    def fit(self, X, y):
        t0 = time.perf_counter()
        self.pipeline_ = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("topgen", TopGenTransformer(random_state=self.random_state, **self.topgen_kwargs)),
                            (
                                "catch22",
                                Pipeline(
                                    [
                                        ("reshape", UnivariateToCollection()),
                                        ("catch22", Catch22()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                ("rf", RandomForestClassifier(random_state=self.random_state, **self.rf_kwargs)),
            ]
        )
        self.pipeline_.fit(X, y)
        self.timings_["fit_total"] = time.perf_counter() - t0
        self.classes_ = self.pipeline_.named_steps["rf"].classes_
        return self

    def predict(self, X):
        t0 = time.perf_counter()
        y_pred = self.pipeline_.predict(X)
        self.timings_["predict_total"] = time.perf_counter() - t0
        return y_pred


# --- evaluation --------------------------------------------------------------


def _method_factory(
    method_name: str,
    seed: int,
    topgen_kwargs: dict,
    rf_kwargs: dict,
):
    if method_name == "TopGen":
        return TopGenRFClassifier(seed, topgen_kwargs, rf_kwargs)
    if method_name == "catch22":
        return AeonCatch22Pipeline(seed, rf_kwargs)
    if method_name == "FreshPRINCE":
        return FreshPRINCEBaseline(seed, rf_kwargs)
    if method_name == "TopGen+catch22":
        return TopGenAddonClassifier(seed, topgen_kwargs, rf_kwargs)
    raise ValueError(f"Unknown method: {method_name}")


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
        "topgen_fit",
        "topgen_fit_self_densities",
        "topgen_fit_class_clouds",
        "topgen_transform_train",
        "topgen_transform_test",
        "topgen_transform_test_cross_persistence",
        "rf_fit",
        "rf_predict",
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

        save_feature_importances(estimator, dataset, method_name, seed)

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


def save_feature_importances(estimator, dataset: str, method_name: str, seed: int) -> None:
    """Dump RandomForest importances aligned with TopGen feature names (Fix 5)."""
    importances = getattr(estimator, "feature_importances_", None)
    names = getattr(estimator, "feature_names_", None)
    if importances is None or names is None:
        return
    os.makedirs(IMPORTANCE_DIR, exist_ok=True)
    path = os.path.join(IMPORTANCE_DIR, f"{dataset}_{method_name}_seed{seed}.json")
    ranked = sorted(
        ({"feature": n, "importance": float(v)} for n, v in zip(names, importances)),
        key=lambda item: item["importance"],
        reverse=True,
    )
    with open(path, "w") as handle:
        json.dump(ranked, handle, indent=2)


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
        help="GunPoint only, seed 0, TopGen+catch22, holdout only, MTD reps, RF n_estimators=50",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: results/accuracy_table_quick.csv or accuracy_table.csv)",
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
        methods = ("TopGen", "catch22", "FreshPRINCE", "TopGen+catch22")
        topgen_kwargs = TOPGEN_KWARGS
        rf_kwargs = RF_KWARGS
        run_cv = True
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
