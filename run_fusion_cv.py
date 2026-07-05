#!/usr/bin/env python3
"""Fusion CV runner: OOF CAWPE weights + stacking matrix from one train-only CV pass."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import run_experiments as exp
from run_modular_cawpe_experiments import cawpe_weights

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


OUTPUT_CSV = "results/accuracy_table_fusion_cv.csv"
CAWPE_WEIGHTS_CSV = "results/fusion_cv/cawpe_weights.csv"
OOF_STACK_DIR = "results/fusion_cv/oof_stack"

COMBO_METHODS = ("TopGen+catch22", "TopGen+TSFresh")
SOLO_MODULES = ("topgen", "catch22", "tsfresh")
METHOD_PREFIX_CAWPE = "CAWPE-cv:"
METHOD_PREFIX_STACKING_LR = "Stacking:LR:"
METHOD_PREFIX_STACKING_HGB = "Stacking:HGB:"
DEFAULT_CAWPE_ALPHA = 4.0

CAWPE_WEIGHT_FIELDS = (
    "dataset",
    "seed",
    "combo",
    "module",
    "oof_acc",
    "weight",
    "cawpe_alpha",
)


def build_stacking_matrix(
    oof_proba: dict[str, np.ndarray],
    module_names: tuple[str, ...],
) -> np.ndarray:
    """Horizontal stack of per-module OOF probabilities."""
    return np.hstack([oof_proba[name] for name in module_names])


def collect_modular_oof(
    train_X: np.ndarray,
    train_y: np.ndarray,
    module_names: tuple[str, ...],
    seed: int,
    topgen_kwargs: dict,
    clf_kwargs: dict,
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray, dict[str, float]]:
    """One StratifiedKFold pass on train: per-module OOF predict_proba."""
    n_train = train_y.shape[0]
    classes = np.sort(np.unique(train_y))
    n_classes = classes.shape[0]
    oof_proba = {name: np.zeros((n_train, n_classes), dtype=float) for name in module_names}
    cv_timings = {"cv_feature_total": 0.0, "cv_classifier_fit": 0.0}

    cv = StratifiedKFold(n_splits=exp.CV_FOLDS, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_X, train_y), start=1):
        print(f"    CV fold {fold_idx}/{exp.CV_FOLDS}: computing feature generators once")
        fold_blocks = exp.compute_feature_blocks(
            module_names,
            train_X[train_idx],
            train_y[train_idx],
            train_X[val_idx],
            seed,
            topgen_kwargs,
        )
        for name in module_names:
            cv_timings["cv_feature_total"] += fold_blocks[name].timings.get(f"{name}_elapsed", 0.0)

        t0 = time.perf_counter()
        for name in module_names:
            block = fold_blocks[name]
            clf = exp.make_classifier(seed, clf_kwargs)
            clf.fit(block.train, train_y[train_idx])
            oof_proba[name][val_idx] = clf.predict_proba(block.eval)
        cv_timings["cv_classifier_fit"] += time.perf_counter() - t0

    oof_acc: dict[str, float] = {}
    for name in module_names:
        y_pred = classes[np.argmax(oof_proba[name], axis=1)]
        oof_acc[name] = float(accuracy_score(train_y, y_pred))

    return oof_proba, oof_acc, classes, cv_timings


def fuse_cawpe_proba(
    oof_proba: dict[str, np.ndarray],
    module_names: tuple[str, ...],
    module_accuracies: dict[str, float],
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    weights, weight_by_module = cawpe_weights(module_names, module_accuracies, alpha)
    fused = np.zeros_like(oof_proba[module_names[0]], dtype=float)
    for weight, name in zip(weights, module_names):
        fused += weight * oof_proba[name]
    return fused, weight_by_module


def make_meta_learner(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, random_state=seed
        )
    raise ValueError(f"Unknown meta-learner: {name!r}")


def fit_meta_learner(
    stack_matrix: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    meta_name: str,
) -> tuple[object, float]:
    t0 = time.perf_counter()
    meta = make_meta_learner(meta_name, seed)
    meta.fit(stack_matrix, y_train)
    return meta, time.perf_counter() - t0


def reset_cawpe_weights_csv(path: str = CAWPE_WEIGHTS_CSV) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAWPE_WEIGHT_FIELDS)
        writer.writeheader()


def append_cawpe_weights(
    dataset: str,
    seed: int,
    combo: str,
    module_names: tuple[str, ...],
    oof_acc: dict[str, float],
    weight_by_module: dict[str, float],
    alpha: float,
    path: str = CAWPE_WEIGHTS_CSV,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAWPE_WEIGHT_FIELDS)
        if write_header:
            writer.writeheader()
        for module in module_names:
            writer.writerow(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "combo": combo,
                    "module": module,
                    "oof_acc": round(oof_acc[module], 6),
                    "weight": round(weight_by_module[module], 6),
                    "cawpe_alpha": alpha,
                }
            )


def save_oof_stack(
    dataset: str,
    seed: int,
    y_train: np.ndarray,
    classes: np.ndarray,
    module_names: tuple[str, ...],
    oof_proba: dict[str, np.ndarray],
    stack_matrix: np.ndarray,
    out_dir: str = OOF_STACK_DIR,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}_seed{seed}.npz")
    payload = {
        "y_train": y_train,
        "classes": classes,
        "module_names": np.asarray(module_names, dtype=object),
        "stack_matrix": stack_matrix,
    }
    for name in module_names:
        payload[f"oof_proba_{name}"] = oof_proba[name]
    np.savez(path, **payload)
    return path


def load_cawpe_cv_weights(
    csv_path: str = CAWPE_WEIGHTS_CSV,
) -> dict[tuple[str, int, str, str], float]:
    """Map (dataset, seed, combo, module) -> normalized CAWPE weight."""
    weights: dict[tuple[str, int, str, str], float] = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], int(row["seed"]), row["combo"], row["module"])
            weights[key] = float(row["weight"])
    return weights


def evaluate_holdout_fusion(
    feature_blocks: dict[str, exp.FeatureBlock],
    module_names: tuple[str, ...],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    clf_kwargs: dict,
    *,
    cawpe_weights_by_module: dict[str, float] | None = None,
    meta_learner=None,
) -> tuple[float, dict[str, float]]:
    """Retrain solo RFs on full train; fuse test probas with CAWPE or meta."""
    t0 = time.perf_counter()
    probas = []
    for name in module_names:
        block = feature_blocks[name]
        clf = exp.make_classifier(seed, clf_kwargs)
        clf.fit(block.train, y_train)
        probas.append(clf.predict_proba(block.eval))
    classifier_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cawpe_weights_by_module is not None:
        fused = np.zeros_like(probas[0], dtype=float)
        for name, proba in zip(module_names, probas):
            fused += cawpe_weights_by_module[name] * proba
        classes = np.sort(np.unique(y_train))
        y_pred = classes[np.argmax(fused, axis=1)]
    elif meta_learner is not None:
        test_stack = np.hstack(probas)
        y_pred = meta_learner.predict(test_stack)
    else:
        raise ValueError("Provide cawpe_weights_by_module or meta_learner")
    classifier_predict = time.perf_counter() - t0
    acc = float(accuracy_score(y_eval, y_pred))
    timings = {
        "classifier_fit": classifier_fit,
        "classifier_predict": classifier_predict,
        "classifier_total": classifier_fit + classifier_predict,
        "n_modules": float(len(module_names)),
    }
    return acc, timings


def _timings_for_fusion_method(
    method_name: str,
    feature_blocks: dict[str, exp.FeatureBlock],
    clf_timings: dict[str, float],
    extra: dict[str, float],
) -> dict[str, float]:
    timings = exp._feature_timings_for_method(method_name, feature_blocks)
    timings.update(clf_timings)
    timings.update(extra)
    feature_fit = timings.get("feature_fit_total", 0.0)
    feature_transform = timings.get("feature_transform_total", 0.0)
    classifier_fit = clf_timings["classifier_fit"]
    classifier_predict = clf_timings["classifier_predict"]
    timings["fit_total"] = feature_fit + classifier_fit
    timings["predict_total"] = feature_transform + classifier_predict
    timings["total"] = timings["fit_total"] + timings["predict_total"]
    return timings


def run_fusion_cv_experiments(
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    topgen_kwargs: dict,
    clf_kwargs: dict,
    alpha: float,
    meta_learners: tuple[str, ...],
    output_csv: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_cache: dict[str, tuple[np.ndarray, ...]] = {}
    split_jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    last_dataset = None
    reset_cawpe_weights_csv()

    for dataset, seed in tqdm(split_jobs, desc="splits", unit="split"):
        if dataset not in data_cache:
            data_cache[dataset] = exp.load_ucr(dataset)
        train_X, train_y, test_X, test_y = data_cache[dataset]
        n_classes = len(np.unique(train_y))

        if dataset != last_dataset:
            print(f"\n{dataset}: train {train_X.shape}, test {test_X.shape}, classes={n_classes}")
            last_dataset = dataset

        print(f"  seed={seed}: CV on train only ({', '.join(SOLO_MODULES)})")
        oof_proba, oof_acc, classes, cv_timings = collect_modular_oof(
            train_X, train_y, SOLO_MODULES, seed, topgen_kwargs, clf_kwargs
        )
        stack_all = build_stacking_matrix(oof_proba, SOLO_MODULES)
        save_oof_stack(dataset, seed, train_y, classes, SOLO_MODULES, oof_proba, stack_all)

        print(f"  seed={seed}: holdout feature generators ({', '.join(SOLO_MODULES)})")
        feature_blocks = exp.compute_feature_blocks(
            SOLO_MODULES, train_X, train_y, test_X, seed, topgen_kwargs
        )

        for method_name in methods:
            module_names = exp.EXPERIMENTS[method_name]
            module_oof_acc = {name: oof_acc[name] for name in module_names}
            module_oof_proba = {name: oof_proba[name] for name in module_names}
            stack_matrix = build_stacking_matrix(module_oof_proba, module_names)

            fused_oof, weight_by_module = fuse_cawpe_proba(
                module_oof_proba, module_names, module_oof_acc, alpha
            )
            append_cawpe_weights(
                dataset, seed, method_name, module_names, module_oof_acc, weight_by_module, alpha
            )
            y_pred_cawpe_oof = classes[np.argmax(fused_oof, axis=1)]
            cv_acc_cawpe = float(accuracy_score(train_y, y_pred_cawpe_oof))

            cawpe_timings_extra = {
                **cv_timings,
                "cawpe_alpha": alpha,
                "cv_oof_accuracy": cv_acc_cawpe,
            }
            for name in module_names:
                cawpe_timings_extra[f"cawpe_oof_acc_{name}"] = module_oof_acc[name]
                cawpe_timings_extra[f"cawpe_weight_{name}"] = weight_by_module[name]

            holdout_cawpe, clf_timings = evaluate_holdout_fusion(
                feature_blocks,
                module_names,
                train_y,
                test_y,
                seed,
                clf_kwargs,
                cawpe_weights_by_module=weight_by_module,
            )
            timings = _timings_for_fusion_method(
                method_name, feature_blocks, clf_timings, cawpe_timings_extra
            )
            csv_method = f"{METHOD_PREFIX_CAWPE}{method_name}"
            row = {
                "dataset": dataset,
                "dataset_type": exp.DATASET_TYPES.get(dataset, "UNKNOWN"),
                "is_dynamical": exp.is_dynamical(dataset),
                "method": csv_method,
                "seed": seed,
                "holdout_accuracy": holdout_cawpe,
                "cv_accuracy": cv_acc_cawpe,
                "time_total_s": round(timings.get("total", 0.0), 2),
                "time_fit_s": round(timings.get("fit_total", 0.0), 2),
                "time_predict_s": round(timings.get("predict_total", 0.0), 2),
                "time_detail_json": json.dumps({k: round(v, 3) for k, v in sorted(timings.items())}),
            }
            rows.append(row)
            weight_str = ", ".join(f"{name}={weight_by_module[name]:.3f}" for name in module_names)
            print(
                f"  {csv_method:32s} seed={seed}  holdout={holdout_cawpe:.4f}"
                f"  cv_oof={cv_acc_cawpe:.4f}  weights: {weight_str}"
            )
            if output_csv is not None:
                exp.write_csv(rows, output_csv, quiet=True)

            for meta_name in meta_learners:
                meta, meta_fit_time = fit_meta_learner(stack_matrix, train_y, seed, meta_name)
                y_pred_stack_oof = meta.predict(stack_matrix)
                cv_acc_stack = float(accuracy_score(train_y, y_pred_stack_oof))
                holdout_stack, clf_timings = evaluate_holdout_fusion(
                    feature_blocks,
                    module_names,
                    train_y,
                    test_y,
                    seed,
                    clf_kwargs,
                    meta_learner=meta,
                )
                prefix = (
                    METHOD_PREFIX_STACKING_LR if meta_name == "lr" else METHOD_PREFIX_STACKING_HGB
                )
                stack_timings_extra = {
                    **cv_timings,
                    "cv_oof_accuracy": cv_acc_stack,
                    "meta_learner_fit": meta_fit_time,
                }
                stack_timings_extra["meta_learner_lr"] = float(meta_name == "lr")
                stack_timings_extra["meta_learner_hgb"] = float(meta_name == "hgb")
                timings = _timings_for_fusion_method(
                    method_name, feature_blocks, clf_timings, stack_timings_extra
                )
                csv_method = f"{prefix}{method_name}"
                row = {
                    "dataset": dataset,
                    "dataset_type": exp.DATASET_TYPES.get(dataset, "UNKNOWN"),
                    "is_dynamical": exp.is_dynamical(dataset),
                    "method": csv_method,
                    "seed": seed,
                    "holdout_accuracy": holdout_stack,
                    "cv_accuracy": cv_acc_stack,
                    "time_total_s": round(timings.get("total", 0.0), 2),
                    "time_fit_s": round(timings.get("fit_total", 0.0), 2),
                    "time_predict_s": round(timings.get("predict_total", 0.0), 2),
                    "time_detail_json": json.dumps(
                        {k: round(v, 3) for k, v in sorted(timings.items())}
                    ),
                }
                rows.append(row)
                print(
                    f"  {csv_method:32s} seed={seed}  holdout={holdout_stack:.4f}"
                    f"  cv_oof={cv_acc_stack:.4f}"
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


def _parse_meta_learners(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("lr", "hgb")
    if value in {"lr", "hgb"}:
        return (value,)
    raise ValueError(f"Unknown --meta-learner {value!r} (expected lr, hgb, or both)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fusion CV: OOF CAWPE weights + stacking matrix from one train-only CV pass"
    )
    parser.add_argument("--quick", action="store_true", help="GunPoint, seed 0, TopGen+catch22")
    parser.add_argument("--lite", action="store_true", help="Use LITE_TOPGEN_KWARGS")
    parser.add_argument("--output", default=None, help=f"Accuracy CSV (default: {OUTPUT_CSV})")
    parser.add_argument(
        "--cawpe-alpha", type=float, default=DEFAULT_CAWPE_ALPHA, help="CAWPE exponent alpha"
    )
    parser.add_argument(
        "--meta-learner",
        choices=("lr", "hgb", "both"),
        default="lr",
        help="Stacking meta-learner (default: lr)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, metavar="SEED")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="DATASET")
    parser.add_argument("--methods", nargs="+", default=None, metavar="METHOD")
    args = parser.parse_args()

    meta_learners = _parse_meta_learners(args.meta_learner)

    if args.quick:
        datasets = exp.QUICK_DATASETS
        seeds = tuple(args.seeds) if args.seeds is not None else exp.QUICK_SEEDS
        methods = _validate_methods(
            tuple(args.methods) if args.methods else ("TopGen+catch22",)
        )
        topgen_kwargs = exp.LITE_TOPGEN_KWARGS if args.lite else exp.QUICK_TOPGEN_KWARGS
        clf_kwargs = exp.QUICK_RF_KWARGS
        output = args.output or "results/accuracy_table_fusion_cv_quick.csv"
    else:
        base_datasets = (
            exp.REPORT_DATASETS + exp.TUNING_DATASETS if args.all_datasets else exp.DATASETS
        )
        if args.datasets is not None:
            unknown = [name for name in args.datasets if name not in exp.DATASET_TYPES]
            if unknown:
                raise ValueError("Unknown dataset(s): " + ", ".join(unknown))
            datasets = tuple(args.datasets)
        else:
            datasets = base_datasets
        seeds = tuple(args.seeds) if args.seeds is not None else exp.SEEDS
        methods = _validate_methods(tuple(args.methods) if args.methods else COMBO_METHODS)
        topgen_kwargs = exp.LITE_TOPGEN_KWARGS if args.lite else exp.TOPGEN_KWARGS
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
        run_cv=True,
    )
    print(
        f"Fusion CV: train-only OOF -> {CAWPE_WEIGHTS_CSV} + {OOF_STACK_DIR}/; "
        f"alpha={args.cawpe_alpha}; meta={','.join(meta_learners)}"
    )

    rows = run_fusion_cv_experiments(
        datasets,
        seeds,
        methods,
        topgen_kwargs,
        clf_kwargs,
        args.cawpe_alpha,
        meta_learners,
        output,
    )
    exp.write_csv(rows, output)
    exp.summarize_by_type(rows)


if __name__ == "__main__":
    main()
