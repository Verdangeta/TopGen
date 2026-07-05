# AGENTS.md

## Cursor Cloud specific instructions

TopGen v2 is a **CLI-driven Python research library** for time-series
classification (topological + baseline features → RandomForest). There is **no
server, web app, or database** — you exercise it by running the `run_*.py`
scripts (see `README.md` for the canonical commands).

### GPU requirement (biggest gotcha)
The core **TopGen cross-barcode features require a CUDA GPU**. `topgen/features.py`
calls MTopDiv (`mtd.calc_cross_barcodes`), which uses `ripserplusplus` (GPU-only,
needs `nvcc`) and hardcodes `pdist_device="cuda"` across the runners. On a
standard CPU-only Cloud VM this means:
- `ripserplusplus` will **not build/install** (no `nvcc`, no GPU).
- `run_minimal_demo.py`, `run_experiments.py` (its default method set includes
  `TopGen`), and the modular runners **cannot complete** — they run all the CPU
  stages (Takens embedding, PCA, class-cloud construction) and then fail at the
  cross-barcode call with `Torch not compiled with CUDA enabled`.
- The `mtd` package itself is pure-Python and installs fine (`pip install` from
  https://github.com/IlyaTrofimov/MTopDiv), and a CPU `torch` wheel installs
  fine, but `import mtd` still fails without `ripserplusplus`.

### What runs on CPU (no GPU)
- The **baseline classification pipeline**: `catch22` / `TSFresh` feature
  generators + `RandomForest` (from `run_experiments.py`). `run_experiments.py`
  has no CLI flag to skip the `TopGen` method, so to run baselines only, drive
  the module functions directly, e.g. `load_ucr`, `compute_feature_blocks(("catch22",), ...)`,
  `combine_method_features`, `fit_predict_from_features`.
- Takens embedding (`gtda.time_series`), PCA, and cloud construction
  (`topgen/clouds.py`).
- UCR datasets are downloaded at runtime from `timeseriesclassification.com`, so
  network access is required.

### Dependency install gotcha (handled by the update script)
`pip install -r requirements.txt` **fails**: `giotto-tda==0.6.2` hard-pins
`scikit-learn==1.3.2`, which conflicts with `scikit-learn>=1.6`. Only
`gtda.time_series` embeddings are used, and they work with modern scikit-learn,
so the fix (already in the startup update script) is to install the main deps
first, then `pip install --no-deps giotto-tda`, then its runtime extras
(`plotly python-igraph pyflagser`). Packages install to the user site
(`~/.local`); no virtualenv activation is needed.

### Lint / tests
There is **no configured linter and no test suite**. Use
`python3 -m py_compile topgen/*.py run_*.py` as the compile/lint gate.

### Repo state note
The `main` branch currently contains only `README.md`/`LICENSE`/`.gitignore`;
the actual code lives on the `cursor/topgen-*` feature branches (the
`topgen-fixes-8403` line is the most complete). Base environment/dev work on the
branch that actually contains the source.
