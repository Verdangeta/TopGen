#!/usr/bin/env python3
"""TopGen v2 experiment runner: UCR datasets, baselines, accuracy CSV."""

from __future__ import annotations

import csv
import os
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

# --- experiment configuration ------------------------------------------------

DATASETS = ("GunPoint", "Coffee", "ItalyPowerDemand")
SEEDS = (0, 1, 2, 3, 4)
CV_FOLDS = 5

UCR_BASE_URL = "https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
OUTPUT_CSV = "results/accuracy_table.csv"

# Shared TopGen settings (short UCR series; fixed embedding like the GunPoint demo).
TOPGEN_KWARGS = dict(
    rep_names=ALL_REP_NAMES,
    hom_dims=(0, 1),
    blocks=("b1", "b2", "b3"),
    class_mode="A",
    search_embedding=False,
    embedding_dimension=10,
    embedding_time_delay=4,
    per_series_budget=30,
    query_size=15,
    class_subcloud_size=20,
    density_samples=15,
    stride=3,
    pdist_device="cuda",
)

RF_KWARGS = dict(n_estimators=200, n_jobs=-1)


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


# --- sklearn adapters for aeon components ------------------------------------


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

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X, y):
        self.pipeline_ = Pipeline(
            [
                ("reshape", UnivariateToCollection()),
                ("catch22", Catch22()),
                ("rf", RandomForestClassifier(random_state=self.random_state, **RF_KWARGS)),
            ]
        )
        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.named_steps["rf"].classes_
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)


class FreshPRINCEBaseline(BaseEstimator, ClassifierMixin):
    """FreshPRINCE classifier baseline (aeon)."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator_ = FreshPRINCEClassifier(
            n_estimators=RF_KWARGS["n_estimators"],
            random_state=self.random_state,
            n_jobs=RF_KWARGS["n_jobs"],
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


class TopGenAddonClassifier(BaseEstimator, ClassifierMixin):
    """TopGen + catch22 via FeatureUnion, then random forest."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X, y):
        topgen = TopGenTransformer(random_state=self.random_state, **TOPGEN_KWARGS)
        self.pipeline_ = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("topgen", topgen),
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
                ("rf", RandomForestClassifier(random_state=self.random_state, **RF_KWARGS)),
            ]
        )
        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.named_steps["rf"].classes_
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)


# --- evaluation --------------------------------------------------------------


def _topgen_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            (
                "topgen",
                TopGenTransformer(random_state=random_state, **TOPGEN_KWARGS),
            ),
            ("rf", RandomForestClassifier(random_state=random_state, **RF_KWARGS)),
        ]
    )


def evaluate_holdout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator,
) -> float:
    estimator.fit(X_train, y_train)
    return float(accuracy_score(y_test, estimator.predict(X_test)))


def evaluate_cv(X: np.ndarray, y: np.ndarray, estimator, seed: int) -> float:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(np.mean(scores))


def run_experiments() -> list[dict[str, object]]:
    methods = {
        "TopGen": lambda seed: _topgen_pipeline(seed),
        "catch22": lambda seed: AeonCatch22Pipeline(random_state=seed),
        "FreshPRINCE": lambda seed: FreshPRINCEBaseline(random_state=seed),
        "TopGen+catch22": lambda seed: TopGenAddonClassifier(random_state=seed),
    }

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        train_X, train_y, test_X, test_y = load_ucr(dataset)
        X_all = np.vstack([train_X, test_X])
        y_all = np.concatenate([train_y, test_y])
        print(f"\n{dataset}: train {train_X.shape}, test {test_X.shape}")

        for seed in SEEDS:
            for method_name, factory in methods.items():
                estimator = factory(seed)
                holdout_acc = evaluate_holdout(train_X, train_y, test_X, test_y, estimator)
                cv_acc = evaluate_cv(X_all, y_all, factory(seed), seed)
                row = {
                    "dataset": dataset,
                    "method": method_name,
                    "seed": seed,
                    "holdout_accuracy": holdout_acc,
                    "cv_accuracy": cv_acc,
                }
                rows.append(row)
                print(
                    f"  {method_name:16s} seed={seed}  "
                    f"holdout={holdout_acc:.4f}  cv={cv_acc:.4f}"
                )
    return rows


def write_csv(rows: list[dict[str, object]], path: str = OUTPUT_CSV) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["dataset", "method", "seed", "holdout_accuracy", "cv_accuracy"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


def main() -> None:
    rows = run_experiments()
    write_csv(rows)


if __name__ == "__main__":
    main()
