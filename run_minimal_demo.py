#!/usr/bin/env python3
"""Minimal TopGen v2 demo: R={MTD}, Mode A, H0+H1 on one UCR dataset."""

from __future__ import annotations

import os
import zipfile
from io import BytesIO
from urllib.request import urlopen

import numpy as np

from topgen.topgen import TopGenTransformer

UCR_ZIP_URL = (
    "https://timeseriesclassification.com/aeon-toolkit/GunPoint.zip"
)


def load_ucr_gunpoint(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download and load the GunPoint train/test split."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "GunPoint.zip")
    if not os.path.exists(zip_path):
        with urlopen(UCR_ZIP_URL) as response:
            zip_path = os.path.join(data_dir, "GunPoint.zip")
            with open(zip_path, "wb") as handle:
                handle.write(response.read())

    with zipfile.ZipFile(zip_path) as archive:
        train_X, train_y = _read_ucr_split(archive, "GunPoint_TRAIN.txt")
        test_X, test_y = _read_ucr_split(archive, "GunPoint_TEST.txt")
    return train_X, train_y, test_X, test_y


def _read_ucr_split(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    with archive.open(member) as handle:
        rows = np.loadtxt(handle)
    labels = rows[:, 0].astype(int)
    series = rows[:, 1:]
    return series, labels


def main() -> None:
    train_X, train_y, test_X, test_y = load_ucr_gunpoint()
    print(f"GunPoint train: {train_X.shape}, test: {test_X.shape}, classes: {np.unique(train_y)}")

    transformer = TopGenTransformer(
        rep_names=("mtd",),
        hom_dims=(0, 1),
        blocks=("b1", "b2", "b3"),
        class_mode="A",
        search_embedding=True,
        per_series_budget=30,
        query_size=15,
        class_subcloud_size=20,
        density_samples=15,
        stride=3,
        random_state=0,
        pdist_device="cpu",
    )

    transformer.fit(train_X, train_y)
    train_features = transformer.transform(train_X, y=train_y)
    test_features = transformer.transform(test_X)

    print(f"Feature matrix shape (train): {train_features.shape}")
    print(f"Feature matrix shape (test):  {test_features.shape}")
    print(f"Features per class: {train_features.shape[1] // len(transformer.classes_)}")
    print(f"Feature names (first 8): {list(transformer.get_feature_names_out()[:8])}")
    print()
    print("Train feature matrix (first 3 rows, rounded):")
    print(np.round(train_features[:3], 4))
    print()
    print("Column means (train):")
    print(np.round(train_features.mean(axis=0), 4))


if __name__ == "__main__":
    main()
