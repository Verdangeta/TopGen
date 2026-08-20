import run_experiments as exp

OUTPUT = "results/accuracy_table_topgen_holdout.csv"
METHODS = ("TopGen",)
SEEDS = (0, 1)
DATASETS = exp.REPORT_DATASETS + exp.TUNING_DATASETS

train_X, train_y, test_X, test_y = exp.load_ucr(DATASETS[0])
exp.explain_runtime_cost(
    n_train=train_X.shape[0],
    n_test=test_X.shape[0],
    n_classes=len(set(train_y)),
    topgen_kwargs=exp.TOPGEN_KWARGS,
    n_datasets=len(DATASETS),
    n_seeds=len(SEEDS),
    n_methods=len(METHODS),
    run_cv=False,
)

rows = exp.run_experiments(
    DATASETS,
    SEEDS,
    METHODS,
    exp.TOPGEN_KWARGS,
    exp.RF_KWARGS,
    run_cv=False,
    output_csv=OUTPUT,
)
exp.write_csv(rows, OUTPUT)
exp.summarize_by_type(rows)
