#!/usr/bin/env python3
"""Build partial independence CSV from run_independence_full.log (in-progress runs)."""

from __future__ import annotations

import argparse
import csv
import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import run_experiments as exp
DEFAULT_LOG = BASE / "results" / "run_independence_full.log"
DEFAULT_OUTPUT = BASE / "results" / "independence" / "independence_full_partial.csv"

FIELDS = (
    "dataset",
    "dataset_type",
    "is_dynamical",
    "block",
    "partner",
    "metric",
    "value",
    "null_mean",
    "p_value",
    "n_seeds",
    "seeds_complete",
)

METRIC_RE = re.compile(
    r"^\s+(b[123]) vs (catch22|tsfresh) (explained_r2|residual_r2|cca_mean_rho): "
    r"value=([\d.]+) null_mean=([\d.]+) p=([\d.]+)"
)
DATASET_RE = re.compile(r"^([A-Za-z0-9]+): train \d+ samples")
SEED_RE = re.compile(r"^\s+seed=(\d+): topgen \+ (catch22|tsfresh)")


def parse_log(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current_dataset: str | None = None
    current_seed: int | None = None

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if "samples (feature-feature, no labels)" in line:
                match = DATASET_RE.match(line.strip())
                if match:
                    current_dataset = match.group(1)
                continue
            seed_match = SEED_RE.match(line)
            if seed_match:
                current_seed = int(seed_match.group(1))
                continue
            metric_match = METRIC_RE.match(line)
            if metric_match and current_dataset is not None and current_seed is not None:
                block, partner, metric, value, null_mean, p_value = metric_match.groups()
                records.append(
                    {
                        "dataset": current_dataset,
                        "seed": current_seed,
                        "block": block,
                        "partner": partner,
                        "metric": metric,
                        "value": float(value),
                        "null_mean": float(null_mean),
                        "p_value": float(p_value),
                    }
                )
    return records


def aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (record["dataset"], record["block"], record["partner"], record["metric"])
        groups[key].append(record)

    rows: list[dict[str, object]] = []
    for (dataset, block, partner, metric), items in sorted(groups.items()):
        seeds = sorted({int(item["seed"]) for item in items})
        rows.append(
            {
                "dataset": dataset,
                "dataset_type": exp.DATASET_TYPES.get(dataset, "UNKNOWN"),
                "is_dynamical": exp.is_dynamical(dataset),
                "block": block,
                "partner": partner,
                "metric": metric,
                "value": round(stats.mean(item["value"] for item in items), 6),
                "null_mean": round(stats.mean(item["null_mean"] for item in items), 6),
                "p_value": round(stats.median(item["p_value"] for item in items), 6),
                "n_seeds": len(seeds),
                "seeds_complete": ",".join(str(seed) for seed in seeds),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export partial independence results from log")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.log.is_file():
        raise FileNotFoundError(f"Log not found: {args.log}")

    records = parse_log(args.log)
    rows = aggregate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    datasets = sorted({row["dataset"] for row in rows})
    splits = len({(r["dataset"], r["seed"]) for r in records})
    print(f"Wrote {args.output}")
    print(f"Parsed metric rows: {len(records)}")
    print(f"Aggregated CSV rows: {len(rows)}")
    print(f"Datasets in CSV: {len(datasets)} -> {', '.join(datasets)}")
    print(f"Approx splits parsed: {splits} / 45")


if __name__ == "__main__":
    main()
