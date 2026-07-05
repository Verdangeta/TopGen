#!/usr/bin/env python3
"""Modular CAWPE fusion: combo methods only, weights from precomputed solo holdout accuracies."""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
from sklearn.metrics import accuracy_score

import run_experiments as exp

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


OUTPUT_CSV = "results/accuracy_table_modular_cawpe.csv"
DEFAULT_SOLO_CSV = "results/accuracy_table_final.csv"
METHOD_PREFIX = "Modular-CAWPE:"
DEFAULT_CAWPE_ALPHA = 4.0

COMBO_METHODS = ("TopGen+catch22", "TopGen+TSFresh")
SOLO_METHODS = ("TopGen", "catch22", "TSFresh")
MODULE_TO_SOLO_METHOD = {
    "topgen": "TopGen",
    "catch22": "catch22",
    "tsfresh": "TSFresh",
}


def load_solo_holdout_accuracies(csv_path: str) -> dict[tuple[str, int, str], float]:
    """Map (dataset, seed, module_name) -> holdout accuracy from solo-method rows."""
    accuracies: dict[tuple[str, int, str], float] = {}
    solo_method_to_module = {method: module for module, method in MODULE_TO_SOLO_METHOD.items()}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in solo_method_to_module:
                continue
            key = (row["dataset"], int(row["seed"]), solo_method_to_module[method])
            accuracies[key] = float(row["holdout_accuracy"])
    return accuracies


def cawpe_weights(
    module_names: tuple[str, ...],
    solo_accuracies: dict[str, float],
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """CAWPE weights w_j = acc_j^alpha, normalized over modules in this ensemble."""
    raw = {name: solo_accuracies[name] ** alpha for name in module_names}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError(f"Non-positive CAWPE weight sum for modules {module_names}: {raw}")
    normalized = {name: value / total for name, value in raw.items()}
    return np.array([normalized[name] for name in module_names]), normalized


def fit_predict_modular_cawpe(
    feature_blocks: dict[str, exp.FeatureBlock],
    module_names: tuple[str, ...],
    solo_accuracies: dict[str, float],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    clf_kwargs: dict,
    alpha: float,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Train one classifier per module; fuse with CAWPE weights from solo holdout accuracies."""
    weights, weight_by_module = cawpe_weights(module_names, solo_accuracies, alpha)

    t0 = time.perf_counter()
    classifiers = []
    probas = []
    for name in module_names:
        block = feature_blocks[name]
        clf = exp.make_classifier(seed, clf_kwargs)
        clf.fit(block.train, y_train)
        classifiers.append(clf)
        probas.append(clf.predict_proba(block.eval))
    classifier_fit = time.perf_counter() - t0

    fused_proba = np.zeros_like(probas[0], dtype=float)
    for weight, proba in zip(weights, probas):
        fused_proba += weight * proba
    classes = classifiers[0].classes_
    y_pred = classes[np.argmax(fused_proba, axis=1)]

    t0 = time.perf_counter()
    acc = float(accuracy_score(y_eval, y_pred))
    classifier_predict = time.perf_counter() - t0

    timings = {
        "classifier_fit": classifier_fit,
        "classifier_predict": classifier_predict,
        "classifier_total": classifier_fit + classifier_predict,
        "n_modules": float(len(module_names)),
        "cawpe_alpha": alpha,
    }
    for name, weight in weight_by_module.items():
        timings[f"cawpe_weight_{name}"] = weight
        timings[f"cawpe_solo_acc_{name}"] = solo_accuracies[name]
    return acc, timings, weight_by_module


def _timings_for_modular_method(
    method_name: str,
    feature_blocks: dict[str, exp.FeatureBlock],
    clf_timings: dict[str, float],
) -> dict[str, float]:
    timings = exp._feature_timings_for_method(method_name, feature_blocks)
    timings.update(clf_timings)
    feature_fit = timings.get("feature_fit_total", 0.0)
    feature_transform = timings.get("feature_transform_total", 0.0)
    classifier_fit = clf_timings["classifier_fit"]
    classifier_predict = clf_timings["classifier_predict"]
    timings["fit_total"] = feature_fit + classifier_fit
    timings["predict_total"] = feature_transform + classifier_predict
    timings["total"] = timings["fit_total"] + timings["predict_total"]
    return timings


def _solo_accuracies_for_split(
    dataset: str,
    seed: int,
    solo_table: dict[tuple[str, int, str], float],
) -> dict[str, float]:
    solo: dict[str, float] = {}
    missing: list[str] = []
    for module, method in MODULE_TO_SOLO_METHOD.items():
        key = (dataset, seed, module)
        if key not in solo_table:
            missing.append(f"{method} ({module})")
            continue
        solo[module] = solo_table[key]
    if missing:
        raise KeyError(f"Missing solo accuracies for {dataset} seed={seed}: {', '.join(missing)}")
    return solo


def run_modular_cawpe_experiments(
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    topgen_kwargs: dict,
    clf_kwargs: dict,
    solo_table: dict[tuple[str, int, str], float],
    alpha: float,
    output_csv: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_cache: dict[str, tuple[np.ndarray, ...]] = {}
    generator_names = exp._unique_generators(methods)
    split_jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    last_dataset = None

    for dataset, seed in tqdm(split_jobs, desc="splits", unit="split"):
        if dataset not in data_cache:
            data_cache[dataset] = exp.load_ucr(dataset)
        train_X, train_y, test_X, test_y = data_cache[dataset]
        n_classes = len(np.unique(train_y))
        solo_accuracies = _solo_accuracies_for_split(dataset, seed, solo_table)

        if dataset != last_dataset:
            print(f"\n{dataset}: train {train_X.shape}, test {test_X.shape}, classes={n_classes}")
            last_dataset = dataset

        print(f"  seed={seed}: computing feature generators once ({', '.join(generator_names)})")
        feature_blocks = exp.compute_feature_blocks(
            generator_names, train_X, train_y, test_X, seed, topgen_kwargs
        )

        for method_name in methods:
            module_names = exp.EXPERIMENTS[method_name]
            module_solo = {name: solo_accuracies[name] for name in module_names}
            holdout_acc, clf_timings, weights = fit_predict_modular_cawpe(
                feature_blocks,
                module_names,
                module_solo,
                train_y,
                test_y,
                seed,
                clf_kwargs,
                alpha,
            )
            timings = _timings_for_modular_method(method_name, feature_blocks, clf_timings)
            csv_method = f"{METHOD_PREFIX}{method_name}"
            weight_str = ", ".join(f"{name}={weights[name]:.3f}" for name in module_names)

            row = {
                "dataset": dataset,
                "dataset_type": exp.DATASET_TYPES.get(dataset, "UNKNOWN"),
                "is_dynamical": exp.is_dynamical(dataset),
                "method": csv_method,
                "seed": seed,
                "holdout_accuracy": holdout_acc,
                "cv_accuracy": float("nan"),
                "time_total_s": round(timings.get("total", 0.0), 2),
                "time_fit_s": round(timings.get("fit_total", 0.0), 2),
                "time_predict_s": round(timings.get("predict_total", 0.0), 2),
                "time_detail_json": json.dumps({k: round(v, 3) for k, v in sorted(timings.items())}),
            }
            rows.append(row)
            print(
                f"  {csv_method:30s} seed={seed}  holdout={holdout_acc:.4f}"
                f"  weights: {weight_str}"
                f"  {exp._format_timings(timings)}"
            )
            if output_csv is not None:
                exp.write_csv(rows, output_csv, quiet=True)
    return rows


def _validate_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    unknown = [name for name in methods if name not in exp.EXPERIMENTS]
    if unknown:
        raise ValueError("Unknown method(s): " + ", ".join(unknown))
    non_combo = [name for name in methods if name not in COMBO_METHODS]
    if non_combo:
        raise ValueError("This runner supports combo methods only: " + ", ".join(non_combo))
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modular CAWPE runner (combo methods; solo accuracies from an existing CSV)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="GunPoint only, seed 0, TopGen+catch22, MTD reps, RF n_estimators=50",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"CSV output path (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--solo-accuracy-csv",
        default=DEFAULT_SOLO_CSV,
        help="CSV with solo TopGen / catch22 / TSFresh holdout accuracies",
    )
    parser.add_argument(
        "--cawpe-alpha",
        type=float,
        default=DEFAULT_CAWPE_ALPHA,
        help="CAWPE exponent alpha (weight_j = solo_acc_j^alpha)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help="Random seeds to run (default: 0 for --quick, 0-4 for full run).",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Include TUNING_DATASETS in addition to REPORT_DATASETS (15 total).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Explicit dataset list to run (overrides --all-datasets/default split).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help=f"Combo methods only (default: {', '.join(COMBO_METHODS)})",
    )
    args = parser.parse_args()

    solo_table = load_solo_holdout_accuracies(args.solo_accuracy_csv)
    if not solo_table:
        raise ValueError(f"No solo accuracies found in {args.solo_accuracy_csv}")

    if args.quick:
        datasets = exp.QUICK_DATASETS
        seeds = tuple(args.seeds) if args.seeds is not None else exp.QUICK_SEEDS
        methods = _validate_methods(
            tuple(args.methods) if args.methods else ("TopGen+catch22",)
        )
        topgen_kwargs = exp.QUICK_TOPGEN_KWARGS
        clf_kwargs = exp.QUICK_RF_KWARGS
        output = args.output or "results/accuracy_table_modular_cawpe_quick.csv"
    else:
        base_datasets = exp.REPORT_DATASETS + exp.TUNING_DATASETS if args.all_datasets else exp.DATASETS
        if args.datasets is not None:
            unknown = [name for name in args.datasets if name not in exp.DATASET_TYPES]
            if unknown:
                raise ValueError("Unknown dataset(s): " + ", ".join(unknown))
            datasets = tuple(args.datasets)
        else:
            datasets = base_datasets
        seeds = tuple(args.seeds) if args.seeds is not None else exp.SEEDS
        methods = _validate_methods(tuple(args.methods) if args.methods else COMBO_METHODS)
        topgen_kwargs = exp.TOPGEN_KWARGS
        clf_kwargs = exp.RF_KWARGS
        output = args.output or OUTPUT_CSV

    train_X, train_y, test_X, test_y = exp.load_ucr(datasets[0])
    exp.explain_runtime_cost(
        n_train=train_X.shape[0],
        n_test=test_X.shape[0],
        n_classes=len(np.unique(train_y)),
        topgen_kwargs=topgen_kwargs,
        n_datasets=len(datasets),
        n_seeds=len(seeds),
        methods=methods,
        run_cv=False,
    )
    print(
        f"Fusion: modular CAWPE (alpha={args.cawpe_alpha}) from {args.solo_accuracy_csv}; "
        f"methods: {', '.join(methods)}"
    )

    rows = run_modular_cawpe_experiments(
        datasets,
        seeds,
        methods,
        topgen_kwargs,
        clf_kwargs,
        solo_table,
        args.cawpe_alpha,
        output,
    )
    exp.write_csv(rows, output)
    exp.summarize_by_type(rows)


if __name__ == "__main__":
    main()
