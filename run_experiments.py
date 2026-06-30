#!/usr/bin/env python3
"""TopGen v2 experiment runner: UCR datasets, baselines, accuracy CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zipfile
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
from aeon.transformations.collection.feature_based import Catch22, TSFresh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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
    methods: tuple[str, ...],
    run_cv: bool,
) -> None:
    """Print cross-barcode budget — dominant cost even on small UCR series."""
    density_repeats = topgen_kwargs.get("density_repeats", 1)
    n_reps = len(topgen_kwargs.get("rep_names", ("mtd",)))
    generators = _unique_generators(methods)
    n_methods = len(methods)
    # LOO fit: every train series is scored vs every class (2 cross-barcodes each);
    # in-class density repeats add (R-1) more query-on-left barcodes per series.
    fit_xbarc = n_train * n_classes * 2 + n_train * max(density_repeats - 1, 0)
    test_xbarc = n_test * n_classes * 2
    per_holdout = fit_xbarc + test_xbarc
    cv_multiplier = 1 + CV_FOLDS if run_cv else 1
    feature_splits = n_datasets * n_seeds * cv_multiplier
    classifier_fits = feature_splits * n_methods
    topgen_passes = feature_splits if "topgen" in generators else 0
    print(
        "Runtime is dominated by GPU cross-barcodes (MTopDiv), not dataset size.\n"
        f"  Per TopGen holdout (this split): ~{per_holdout} cross-barcodes "
        f"({fit_xbarc} LOO fit + {test_xbarc} test),\n"
        f"  {n_reps} linear reps each reuse the same barcode.\n"
        f"  Feature generators per split: {', '.join(generators)}.\n"
        f"  TopGen feature passes after reuse: {topgen_passes}; "
        f"classifier fits: {classifier_fits}.\n"
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


# --- evaluation --------------------------------------------------------------


@dataclass
class FeatureBlock:
    train: np.ndarray
    eval: np.ndarray
    records: list[dict[str, object]]
    timings: dict[str, float]


def _unique_generators(methods: tuple[str, ...]) -> tuple[str, ...]:
    """Generators needed by these methods, in first-use order."""
    names: list[str] = []
    for method in methods:
        if method not in EXPERIMENTS:
            raise ValueError(f"Unknown method: {method}")
        for name in EXPERIMENTS[method]:
            if name not in names:
                names.append(name)
    return tuple(names)


def _as_feature_matrix(matrix, name: str, split: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} produced a {matrix.ndim}D {split} matrix; expected 2D")
    return matrix


def _generator_timings(name: str, generator, fit_elapsed: float, transform_elapsed: float) -> dict[str, float]:
    timings = {
        f"{name}_fit_elapsed": fit_elapsed,
        f"{name}_transform_elapsed": transform_elapsed,
        f"{name}_elapsed": fit_elapsed + transform_elapsed,
    }
    transformer = getattr(generator, "transformer", None)
    for attr, prefix in (("fit_timings_", "fit"), ("transform_timings_", "transform")):
        for key, value in getattr(transformer, attr, {}).items():
            timings[f"{name}_{prefix}_{key}"] = float(value)
    return timings


def compute_feature_blocks(
    generator_names: tuple[str, ...],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
    topgen_kwargs: dict,
) -> dict[str, FeatureBlock]:
    """First cycle: fit each requested generator once and keep its train/eval blocks."""
    blocks: dict[str, FeatureBlock] = {}
    for name in generator_names:
        generator = make_generator(name, seed, topgen_kwargs)

        t0 = time.perf_counter()
        train_block = _as_feature_matrix(generator.fit_transform(X_train, y_train), name, "train")
        fit_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        eval_block = _as_feature_matrix(generator.transform(X_eval), name, "eval")
        transform_elapsed = time.perf_counter() - t0

        records = generator.records()
        if len(records) != train_block.shape[1]:
            raise ValueError(
                f"{name} records ({len(records)}) do not match feature columns "
                f"({train_block.shape[1]})"
            )

        blocks[name] = FeatureBlock(
            train=train_block,
            eval=eval_block,
            records=records,
            timings=_generator_timings(name, generator, fit_elapsed, transform_elapsed),
        )
        print(f"    {name:7s} features: train {train_block.shape}, eval {eval_block.shape}")
    return blocks


def combine_method_features(
    method_name: str,
    feature_blocks: dict[str, FeatureBlock],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Second cycle input: concatenate already-computed blocks for one method."""
    generator_names = EXPERIMENTS[method_name]
    train_parts = [feature_blocks[name].train for name in generator_names]
    eval_parts = [feature_blocks[name].eval for name in generator_names]
    records: list[dict[str, object]] = []
    for name in generator_names:
        records.extend(feature_blocks[name].records)
    train = train_parts[0] if len(train_parts) == 1 else np.hstack(train_parts)
    eval_matrix = eval_parts[0] if len(eval_parts) == 1 else np.hstack(eval_parts)
    return train, eval_matrix, records


def _feature_timings_for_method(
    method_name: str,
    feature_blocks: dict[str, FeatureBlock],
) -> dict[str, float]:
    timings: dict[str, float] = {}
    feature_fit = 0.0
    feature_transform = 0.0
    for name in EXPERIMENTS[method_name]:
        block_timings = feature_blocks[name].timings
        feature_fit += block_timings[f"{name}_fit_elapsed"]
        feature_transform += block_timings[f"{name}_transform_elapsed"]
        timings.update(block_timings)
    timings["feature_fit_total"] = feature_fit
    timings["feature_transform_total"] = feature_transform
    timings["feature_total"] = feature_fit + feature_transform
    timings["features_reused"] = 1.0
    return timings


def fit_predict_from_features(
    X_train_features: np.ndarray,
    y_train: np.ndarray,
    X_eval_features: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    rf_kwargs: dict,
    feature_timings: dict[str, float] | None = None,
) -> tuple[float, RandomForestClassifier, dict[str, float]]:
    """Train the shared classifier on precomputed features and score an eval split."""
    timings = dict(feature_timings or {})

    t0 = time.perf_counter()
    clf = make_classifier(seed, rf_kwargs)
    clf.fit(X_train_features, y_train)
    classifier_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_eval_features)
    classifier_predict = time.perf_counter() - t0

    feature_fit = timings.get("feature_fit_total", 0.0)
    feature_transform = timings.get("feature_transform_total", 0.0)
    timings["classifier_fit"] = classifier_fit
    timings["classifier_predict"] = classifier_predict
    timings["classifier_total"] = classifier_fit + classifier_predict
    timings["fit_total"] = feature_fit + classifier_fit
    timings["predict_total"] = feature_transform + classifier_predict
    timings["total"] = timings["fit_total"] + timings["predict_total"]
    return float(accuracy_score(y_eval, y_pred)), clf, timings


def evaluate_cv_reused(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    methods: tuple[str, ...],
    generator_names: tuple[str, ...],
    topgen_kwargs: dict,
    rf_kwargs: dict,
) -> dict[str, float]:
    """Manual CV so each fold computes every generator once, then reuses blocks."""
    scores = {method: [] for method in methods}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, eval_idx) in enumerate(cv.split(X, y), start=1):
        print(f"    CV fold {fold_idx}/{CV_FOLDS}: computing feature generators once")
        fold_blocks = compute_feature_blocks(
            generator_names,
            X[train_idx],
            y[train_idx],
            X[eval_idx],
            seed,
            topgen_kwargs,
        )
        for method_name in methods:
            X_train_features, X_eval_features, _ = combine_method_features(method_name, fold_blocks)
            acc, _, _ = fit_predict_from_features(
                X_train_features,
                y[train_idx],
                X_eval_features,
                y[eval_idx],
                seed,
                rf_kwargs,
            )
            scores[method_name].append(acc)
    return {method: float(np.mean(method_scores)) for method, method_scores in scores.items()}


def _format_timings(timings: dict[str, float]) -> str:
    parts = []
    for key in (
        "feature_fit_total",
        "fit_total",
        "topgen_fit_class_clouds",
        "topgen_fit_cross_persistence",
        "topgen_fit_self_densities",
        "feature_transform_total",
        "topgen_transform_cross_persistence",
        "classifier_fit",
        "classifier_predict",
        "predict_total",
        "total",
    ):
        if key in timings:
            parts.append(f"{key}={timings[key]:.1f}s")
    return "  ".join(parts)


def _format_cv_acc(cv_acc: float) -> str:
    return f"  cv={cv_acc:.4f}" if not np.isnan(cv_acc) else ""


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
    data_cache: dict[str, tuple[np.ndarray, ...]] = {}
    generator_names = _unique_generators(methods)
    split_jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    last_dataset = None

    for dataset, seed in tqdm(split_jobs, desc="splits", unit="split"):
        if dataset not in data_cache:
            data_cache[dataset] = load_ucr(dataset)
        train_X, train_y, test_X, test_y = data_cache[dataset]
        n_classes = len(np.unique(train_y))

        if dataset != last_dataset:
            print(f"\n{dataset}: train {train_X.shape}, test {test_X.shape}, classes={n_classes}")
            last_dataset = dataset

        print(f"  seed={seed}: computing feature generators once ({', '.join(generator_names)})")
        feature_blocks = compute_feature_blocks(
            generator_names, train_X, train_y, test_X, seed, topgen_kwargs
        )

        cv_accs = {method_name: float("nan") for method_name in methods}
        if run_cv:
            X_all = np.vstack([train_X, test_X])
            y_all = np.concatenate([train_y, test_y])
            cv_accs = evaluate_cv_reused(
                X_all, y_all, seed, methods, generator_names, topgen_kwargs, rf_kwargs
            )

        for method_name in methods:
            X_train_features, X_test_features, feature_records = combine_method_features(
                method_name, feature_blocks
            )
            holdout_acc, clf, timings = fit_predict_from_features(
                X_train_features,
                train_y,
                X_test_features,
                test_y,
                seed,
                rf_kwargs,
                _feature_timings_for_method(method_name, feature_blocks),
            )

            save_feature_importances(
                clf,
                feature_records,
                dataset=dataset,
                seed=seed,
                dataset_type=DATASET_TYPES.get(dataset, "UNKNOWN"),
                is_dynamical=is_dynamical(dataset),
                experiment=method_name,
                out_dir=IMPORTANCE_DIR,
                X_val=X_test_features,
                y_val=test_y,
            )

            cv_acc = cv_accs[method_name]
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
                + _format_cv_acc(cv_acc)
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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help="Random seeds to run (default: 0 for --quick, 0-4 for full run).",
    )
    args = parser.parse_args()

    if args.quick:
        datasets = QUICK_DATASETS
        seeds = tuple(args.seeds) if args.seeds is not None else QUICK_SEEDS
        methods = QUICK_METHODS
        topgen_kwargs = QUICK_TOPGEN_KWARGS
        rf_kwargs = QUICK_RF_KWARGS
        run_cv = False
        output = args.output or "results/accuracy_table_quick.csv"
    else:
        datasets = DATASETS
        seeds = tuple(args.seeds) if args.seeds is not None else SEEDS
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
        methods=methods,
        run_cv=run_cv,
    )

    rows = run_experiments(datasets, seeds, methods, topgen_kwargs, rf_kwargs, run_cv, output)
    write_csv(rows, output)
    summarize_by_type(rows)


if __name__ == "__main__":
    main()
