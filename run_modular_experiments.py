#!/usr/bin/env python3
"""Modular fusion experiments: one classifier per feature block, uniform proba average."""

from __future__ import annotations

import argparse
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


OUTPUT_CSV = "results/accuracy_table_modular_uniform.csv"
OUTPUT_CSV_BY_CLASSIFIER = {
    "rf": OUTPUT_CSV,
    "rotation_forest": "results/accuracy_table_modular_uniform_rotation_forest.csv",
}
METHOD_PREFIX = "Modular:"


def fit_predict_modular_uniform(
    feature_blocks: dict[str, exp.FeatureBlock],
    module_names: tuple[str, ...],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    clf_kwargs: dict,
    classifier_name: str = "rf",
) -> tuple[float, dict[str, float]]:
    """Train one classifier per module; predict via uniform average of predict_proba."""
    t0 = time.perf_counter()
    classifiers = []
    probas = []
    for name in module_names:
        block = feature_blocks[name]
        clf = exp.make_classifier(seed, clf_kwargs, classifier_name)
        clf.fit(block.train, y_train)
        classifiers.append(clf)
        probas.append(clf.predict_proba(block.eval))
    classifier_fit = time.perf_counter() - t0

    avg_proba = np.mean(probas, axis=0)
    classes = classifiers[0].classes_
    y_pred = classes[np.argmax(avg_proba, axis=1)]

    t0 = time.perf_counter()
    acc = float(accuracy_score(y_eval, y_pred))
    classifier_predict = time.perf_counter() - t0

    return acc, {
        "classifier_fit": classifier_fit,
        "classifier_predict": classifier_predict,
        "classifier_total": classifier_fit + classifier_predict,
        "n_modules": float(len(module_names)),
    }


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


def run_modular_experiments(
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    topgen_kwargs: dict,
    clf_kwargs: dict,
    classifier_name: str = "rf",
    output_csv: str | None = None,
) -> list[dict[str, object]]:
    """Holdout evaluation with modular uniform probability fusion."""
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

        if dataset != last_dataset:
            print(f"\n{dataset}: train {train_X.shape}, test {test_X.shape}, classes={n_classes}")
            last_dataset = dataset

        print(f"  seed={seed}: computing feature generators once ({', '.join(generator_names)})")
        feature_blocks = exp.compute_feature_blocks(
            generator_names, train_X, train_y, test_X, seed, topgen_kwargs
        )

        for method_name in methods:
            module_names = exp.EXPERIMENTS[method_name]
            holdout_acc, clf_timings = fit_predict_modular_uniform(
                feature_blocks,
                module_names,
                train_y,
                test_y,
                seed,
                clf_kwargs,
                classifier_name,
            )
            timings = _timings_for_modular_method(method_name, feature_blocks, clf_timings)
            csv_method = f"{METHOD_PREFIX}{method_name}"

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
                f"  {csv_method:24s} seed={seed}  holdout={holdout_acc:.4f}"
                f"  {exp._format_timings(timings)}"
            )
            if output_csv is not None:
                exp.write_csv(rows, output_csv, quiet=True)
    return rows


def _validate_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    unknown = [name for name in methods if name not in exp.EXPERIMENTS]
    if unknown:
        raise ValueError("Unknown method(s): " + ", ".join(unknown))
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modular fusion runner: independent classifiers + uniform proba average"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="GunPoint only, seed 0, TopGen and TopGen+catch22, MTD reps, RF n_estimators=50",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="TopGen lite: H0-only hom_dims=(0,) and rep_names without betti_2.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"CSV output path (default: {OUTPUT_CSV})",
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
        help="Experiment recipes from run_experiments.EXPERIMENTS (default: all).",
    )
    parser.add_argument(
        "--classifier",
        choices=tuple(exp.CLASSIFIER_KWARGS),
        default="rf",
        help="Classifier for each module (default: rf).",
    )
    args = parser.parse_args()

    if args.quick:
        datasets = exp.QUICK_DATASETS
        seeds = tuple(args.seeds) if args.seeds is not None else exp.QUICK_SEEDS
        methods = _validate_methods(tuple(args.methods) if args.methods else exp.QUICK_METHODS)
        topgen_kwargs = exp.QUICK_TOPGEN_KWARGS
        clf_kwargs = exp.QUICK_RF_KWARGS if args.classifier == "rf" else exp.ROTATION_FOREST_KWARGS
        output = args.output or "results/accuracy_table_modular_quick.csv"
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
        methods = _validate_methods(tuple(args.methods) if args.methods else tuple(exp.EXPERIMENTS))
        topgen_kwargs = exp.LITE_TOPGEN_KWARGS if args.lite else exp.TOPGEN_KWARGS
        clf_kwargs = exp.CLASSIFIER_KWARGS[args.classifier]
        output = args.output or (
            "results/accuracy_table_modular_lite.csv"
            if args.lite
            else OUTPUT_CSV_BY_CLASSIFIER[args.classifier]
        )

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
        f"Fusion: modular uniform average ({METHOD_PREFIX}* methods in CSV); "
        f"classifier={args.classifier}"
    )

    rows = run_modular_experiments(
        datasets, seeds, methods, topgen_kwargs, clf_kwargs, args.classifier, output
    )
    exp.write_csv(rows, output)
    exp.summarize_by_type(rows)


if __name__ == "__main__":
    main()
