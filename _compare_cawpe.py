#!/usr/bin/env python3
import csv
import statistics as stats
from collections import defaultdict

BASE = "/home/PD_exp/topgen-v2-minimal-4a60"
OLD_PATH = f"{BASE}/results/accuracy_table_modular_cawpe.csv"
NEW_PATH = f"{BASE}/results/accuracy_table_fusion_cv_lite.csv"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def combo_from_method(method: str) -> str | None:
    for prefix in ("Modular-CAWPE:", "CAWPE-cv:", "Stacking:LR:", "Stacking:HGB:"):
        if method.startswith(prefix):
            return method[len(prefix) :]
    return None


old_rows = load(OLD_PATH)
new_rows = load(NEW_PATH)

by_ds_seed = defaultdict(list)
for r in new_rows:
    by_ds_seed[(r["dataset"], r["seed"])].append(r["method"])

complete_datasets = set()
for ds in {r["dataset"] for r in new_rows}:
    if all(len(by_ds_seed.get((ds, str(s)), [])) == 4 for s in range(3)):
        complete_datasets.add(ds)

partial = sorted({r["dataset"] for r in new_rows} - complete_datasets)

print("Complete datasets (3 seeds x 4 methods):", sorted(complete_datasets))
if partial:
    print("Partial / in progress:", partial)
print()


def agg(rows, method_prefix, col="holdout_accuracy"):
    groups = defaultdict(list)
    for r in rows:
        if r["dataset"] not in complete_datasets:
            continue
        m = r["method"]
        if not m.startswith(method_prefix):
            continue
        combo = combo_from_method(m)
        if combo is None:
            continue
        groups[combo].append(float(r[col]))
    return {k: (stats.mean(v), len(v)) for k, v in sorted(groups.items())}


old_cawpe = agg(old_rows, "Modular-CAWPE:")
new_cawpe_h = agg(new_rows, "CAWPE-cv:")
new_cawpe_cv = agg(new_rows, "CAWPE-cv:", "cv_accuracy")
stack_h = agg(new_rows, "Stacking:LR:")
stack_cv = agg(new_rows, "Stacking:LR:", "cv_accuracy")

print("=== HOLDOUT test accuracy (mean over complete datasets x 3 seeds) ===")
print(f"{'Combo':<22} {'Old CAWPE':>12} {'Real CAWPE':>12} {'Stack LR':>12}")
for combo in sorted(set(old_cawpe) | set(new_cawpe_h) | set(stack_h)):
    o = old_cawpe.get(combo, (float("nan"), 0))[0]
    c = new_cawpe_h.get(combo, (float("nan"), 0))[0]
    s = stack_h.get(combo, (float("nan"), 0))[0]
    print(f"{combo:<22} {o:>12.4f} {c:>12.4f} {s:>12.4f}")

print()
print("=== TRAIN OOF cv_accuracy (fusion CV only) ===")
print(f"{'Combo':<22} {'Real CAWPE':>12} {'Stack LR':>12}")
for combo in sorted(set(new_cawpe_cv) | set(stack_cv)):
    c = new_cawpe_cv.get(combo, (float("nan"), 0))[0]
    s = stack_cv.get(combo, (float("nan"), 0))[0]
    print(f"{combo:<22} {c:>12.4f} {s:>12.4f}")

print()
print("=== Per-dataset holdout (mean over 3 seeds) ===")
print(f"{'Dataset':<22} {'Combo':<18} {'Old':>8} {'Real':>8} {'Stack':>8}")
for ds in sorted(complete_datasets):
    for combo in ("TopGen+catch22", "TopGen+TSFresh"):
        def mean_hold(rows, method):
            vals = [
                float(r["holdout_accuracy"])
                for r in rows
                if r["dataset"] == ds and r["method"] == method
            ]
            return stats.mean(vals) if vals else float("nan")

        o = mean_hold(old_rows, f"Modular-CAWPE:{combo}")
        c = mean_hold(new_rows, f"CAWPE-cv:{combo}")
        s = mean_hold(new_rows, f"Stacking:LR:{combo}")
        print(f"{ds:<22} {combo:<18} {o:>8.3f} {c:>8.3f} {s:>8.3f}")

if partial:
    print()
    print("=== Partial datasets (rows written incrementally) ===")
    print(f"{'Dataset':<22} {'Method':<28} {'seed':>4} {'holdout':>8} {'cv_oof':>8}")
    for r in sorted(new_rows, key=lambda x: (x["dataset"], int(x["seed"]), x["method"])):
        if r["dataset"] in complete_datasets:
            continue
        cv = r["cv_accuracy"]
        cv_s = f"{float(cv):.3f}" if cv and cv != "nan" else "   n/a"
        print(
            f"{r['dataset']:<22} {r['method']:<28} {r['seed']:>4} "
            f"{float(r['holdout_accuracy']):>8.3f} {cv_s:>8}"
        )

n = len(complete_datasets) * 3
print()
print(f"Rows per combo in aggregates: {n}")
print("Old CAWPE = modular_cawpe (FULL TopGen, holdout solo weights).")
print("Real CAWPE = fusion_cv --lite (LITE TopGen, OOF train weights).")
