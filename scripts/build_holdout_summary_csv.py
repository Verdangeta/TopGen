#!/usr/bin/env python3
"""Build one CSV: per-dataset holdout accuracy (mean over seeds) for all methods."""

from __future__ import annotations

import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

BASE = Path("/home/PD_exp/topgen-v2-minimal-4a60")
OUTPUT = BASE / "results" / "accuracy_table_all_methods_holdout.csv"

DATASETS_ORDER = [
    "GunPoint",
    "ECG200",
    "Lightning2",
    "Earthquakes",
    "RefrigerationDevices",
    "WormsTwoClass",
    "OliveOil",
    "Strawberry",
    "ArrowHead",
    "Herring",
    "Coffee",
    "Computers",
    "ItalyPowerDemand",
    "Plane",
    "Worms",
]

# (csv_column_name, source_file, method_name_in_source)
COLUMNS = [
    ("topgen", "accuracy_table_final.csv", "TopGen"),
    ("catch22", "accuracy_table_final.csv", "catch22"),
    ("tsfresh", "accuracy_table_final.csv", "TSFresh"),
    ("stacked_topgen_catch22", "accuracy_table_final.csv", "TopGen+catch22"),
    ("stacked_topgen_tsfresh", "accuracy_table_final.csv", "TopGen+TSFresh"),
    ("modular_topgen_catch22", "accuracy_table_modular_uniform.csv", "Modular:TopGen+catch22"),
    ("modular_topgen_tsfresh", "accuracy_table_modular_uniform.csv", "Modular:TopGen+TSFresh"),
    ("cawpe_old_topgen_catch22", "accuracy_table_modular_cawpe.csv", "Modular-CAWPE:TopGen+catch22"),
    ("cawpe_old_topgen_tsfresh", "accuracy_table_modular_cawpe.csv", "Modular-CAWPE:TopGen+TSFresh"),
    ("cawpe_cv_lite_topgen_catch22", "accuracy_table_fusion_cv_lite.csv", "CAWPE-cv:TopGen+catch22"),
    ("cawpe_cv_lite_topgen_tsfresh", "accuracy_table_fusion_cv_lite.csv", "CAWPE-cv:TopGen+TSFresh"),
    ("cawpe_cv_full_topgen_catch22", "accuracy_table_fusion_cv.csv", "CAWPE-cv:TopGen+catch22"),
    ("cawpe_cv_full_topgen_tsfresh", "accuracy_table_fusion_cv.csv", "CAWPE-cv:TopGen+TSFresh"),
    ("stacking_lr_topgen_catch22", "accuracy_table_fusion_cv_lite.csv", "Stacking:LR:TopGen+catch22"),
    ("stacking_lr_topgen_tsfresh", "accuracy_table_fusion_cv_lite.csv", "Stacking:LR:TopGen+TSFresh"),
]

META_FIELDS = ("dataset", "dataset_type", "is_dynamical")
VALUE_FIELDS = tuple(col for col, _, _ in COLUMNS)
FIELDNAMES = META_FIELDS + VALUE_FIELDS


def load_rows(filename: str) -> list[dict[str, str]]:
    path = BASE / "results" / filename
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_holdout_by_dataset(rows: list[dict[str, str]], method: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            groups[row["dataset"]].append(float(row["holdout_accuracy"]))
    return {dataset: stats.mean(values) for dataset, values in groups.items()}


def main() -> None:
    source_cache: dict[str, list[dict[str, str]]] = {}
    column_data: dict[str, dict[str, float]] = {}
    meta_by_dataset: dict[str, dict[str, str]] = {}

    for col_name, source_file, method in COLUMNS:
        if source_file not in source_cache:
            source_cache[source_file] = load_rows(source_file)
        rows = source_cache[source_file]
        column_data[col_name] = mean_holdout_by_dataset(rows, method)
        for row in rows:
            if row["dataset"] not in meta_by_dataset:
                meta_by_dataset[row["dataset"]] = {
                    "dataset_type": row.get("dataset_type", ""),
                    "is_dynamical": row.get("is_dynamical", ""),
                }

    out_rows: list[dict[str, str | float]] = []
    for dataset in DATASETS_ORDER:
        row: dict[str, str | float] = {
            "dataset": dataset,
            "dataset_type": meta_by_dataset.get(dataset, {}).get("dataset_type", ""),
            "is_dynamical": meta_by_dataset.get(dataset, {}).get("is_dynamical", ""),
        }
        for col_name in VALUE_FIELDS:
            value = column_data[col_name].get(dataset)
            row[col_name] = round(value, 6) if value is not None else ""
        out_rows.append(row)

    mean_row: dict[str, str | float] = {"dataset": "MEAN", "dataset_type": "", "is_dynamical": ""}
    for col_name in VALUE_FIELDS:
        values = [column_data[col_name][ds] for ds in DATASETS_ORDER if ds in column_data[col_name]]
        mean_row[col_name] = round(stats.mean(values), 6) if values else ""
    out_rows.append(mean_row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {OUTPUT}")
    print(f"Rows: {len(out_rows)} ({len(DATASETS_ORDER)} datasets + MEAN)")
    print(f"Columns: {len(VALUE_FIELDS)} method holdout accuracies")


if __name__ == "__main__":
    main()
