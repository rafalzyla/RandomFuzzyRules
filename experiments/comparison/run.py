"""Compare six interpretable classifiers on selected binary UCI datasets.

The benchmark uses the common UCI loader and fold-local preprocessing defined
in ``experiments.utils``. All estimators are evaluated with the same 10-fold
stratified cross-validation splits.

Outputs are written to ``results/comparison``:

- ``fold_results.csv``
- ``errors.csv`` when errors occur
- ``dataset_mean_results.csv``
- ``critical_difference_accuracy.png``
- ``pairwise_accuracy_rfr_vs_gpr.png``
- ``mean_fit_time.png``

The fold-level benchmark can be resumed because combinations already present
in ``fold_results.csv`` are skipped by ``utils.run_benchmark``.
"""

from __future__ import annotations

import argparse
import gc
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from aeon.visualisation import plot_pairwise_scatter

from experiments import utils
from experiments.ablation.validate_default_configuration import SELECTED_DEFAULTS
from experiments.datasets import COMPARISON_DATASETS
from experiments.gpr_fast_bridge import GPRFastSubprocessClassifier
from random_fuzzy_rules import RandomFuzzyRulesClassifier


MODEL_ORDER = [
    "RFR",
    "GPR",
    "FIGS",
    "RuleFit",
    "HSTree",
    "GreedyRuleList",
]

DISPLAY_LABELS = {
    "RFR": "RFR",
    "GPR": "GPR",
    "FIGS": "FIGS",
    "RuleFit": "RuleFit",
    "HSTree": "HSTree",
    "GreedyRuleList": "GreedyRuleList",
}

ALPHA = 0.05


def _supported_kwargs(cls, **kwargs):
    """Retain only constructor arguments explicitly supported by a class."""
    parameters = inspect.signature(cls).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def make_estimators(
    continuous_features,
    categorical_feature_groups,
    transformed_feature_names,
):
    """Create fresh fitted-model candidates expected by utils.run_benchmark."""
    from imodels import (
        FIGSClassifier,
        GreedyRuleListClassifier,
        HSTreeClassifier,
        RuleFitClassifier,
    )


    figs_kwargs = _supported_kwargs(
        FIGSClassifier,
        n_jobs=-1, # All threads are used
        random_state=utils.RANDOM_STATE,
    )
    rulefit_kwargs = _supported_kwargs(
        RuleFitClassifier,
        random_state=utils.RANDOM_STATE,
    )
    hstree_kwargs = _supported_kwargs(
        HSTreeClassifier,
        random_state=utils.RANDOM_STATE,
    )
    greedy_kwargs = _supported_kwargs(
        GreedyRuleListClassifier,
        random_state=utils.RANDOM_STATE,
    )

    return {
        "RFR": RandomFuzzyRulesClassifier(
            **dict(SELECTED_DEFAULTS),
            continuous_features=continuous_features,
            categorical_feature_groups=categorical_feature_groups,
            feature_names=transformed_feature_names,
            random_state=utils.RANDOM_STATE,
        ),
        "GPR": GPRFastSubprocessClassifier(
            feature_names=transformed_feature_names,
            n_populations=100,
            n_generations=100,
            threshold=0.5,
            verbose=False,
            max_n_of_rules=6,
            max_n_of_ands=6,
            base_pb=0.1,
            random_state=utils.RANDOM_STATE,
            n_jobs=None, # All threads are used
        ),
        "FIGS": FIGSClassifier(**figs_kwargs),
        "RuleFit": RuleFitClassifier(**rulefit_kwargs),
        "HSTree": HSTreeClassifier(**hstree_kwargs),
        "GreedyRuleList": GreedyRuleListClassifier(**greedy_kwargs),
    }


def warm_up_estimators():
    """Compile RFR kernels and smoke-test all estimator integrations."""
    rng = np.random.default_rng(utils.RANDOM_STATE)

    # Four numerical columns followed by two valid two-column one-hot groups.
    X_numerical = rng.uniform(0.0, 1.0, size=(100, 4))
    category_a = rng.integers(0, 2, size=100)
    category_b = rng.integers(0, 2, size=100)
    X_categorical = np.column_stack(
        (
            1.0 - category_a,
            category_a,
            1.0 - category_b,
            category_b,
        )
    )
    X = np.ascontiguousarray(
        np.column_stack((X_numerical, X_categorical)),
        dtype=np.float64,
    )
    y = np.array([0, 1] * 50, dtype=np.int64)
    feature_names = [f"x{i + 1}" for i in range(X.shape[1])]

    estimators = make_estimators(
        continuous_features=[0, 1, 2, 3],
        categorical_feature_groups=[(4, 5), (6, 7)],
        transformed_feature_names=feature_names,
    )

    # A 100x100 GPR run would make warm-up unnecessarily expensive. Replace
    # only the smoke-test instance with a tiny configuration. Benchmark fits
    # still use the required 100 populations and 100 generations.
    if hasattr(estimators["GPR"], "close"):
        estimators["GPR"].close()
    estimators["GPR"] = GPRFastSubprocessClassifier(
        feature_names=feature_names,
        n_populations=2,
        n_generations=2,
        threshold=0.5,
        verbose=False,
        max_n_of_rules=2,
        max_n_of_ands=2,
        base_pb=0.1,
        random_state=utils.RANDOM_STATE,
        n_jobs=None, # All threads are used
    )

    # Likewise, use a small RFR candidate budget only for kernel compilation.
    estimators["RFR"] = RandomFuzzyRulesClassifier(
        max_rules=2,
        max_rules_len=2,
        max_literal_repetitions=2,
        threshold=0.5,
        n_candidates=50,
        max_sampling_attempts=500,
        sampling_type_number=1,
        sampling_type_length=2,
        preprocessed=True,
        continuous_features=[0, 1, 2, 3],
        categorical_feature_groups=[(4, 5), (6, 7)],
        feature_names=feature_names,
        random_state=utils.RANDOM_STATE,
    )

    for name in MODEL_ORDER:
        print(f"Warm-up: {name}")
        estimator = estimators[name]
        try:
            estimator.fit(X, y)
            estimator.predict(X[:10])
        finally:
            if hasattr(estimator, "close"):
                estimator.close()
            del estimator
            gc.collect()

    print("Estimator warm-up completed.")


def draw_pairwise_accuracy(dataset_means, OUTPUT_DIR):
    """Compare RFR and GPR dataset-level mean Accuracy values."""
    matrix = utils.metric_matrix(
        dataset_results=dataset_means,
        metric="accuracy",
        estimator_order=["RFR", "GPR"],
        display_labels=None,
    )

    fig, _ = plot_pairwise_scatter(
        results_a=matrix["RFR"].to_numpy(),
        results_b=matrix["GPR"].to_numpy(),
        method_a="RFR",
        method_b="GPR",
        metric="accuracy",
        lower_better=False,
        statistic_tests=True,
        title="RFR versus GPR — Accuracy",
        figsize=(8, 8),
        best_on_top=False,
    )
    output_file = OUTPUT_DIR / "pairwise_accuracy_rfr_vs_gpr.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return matrix


def generate_outputs(OUTPUT_DIR, RESULTS_FILE, DATASET_MEANS_FILE):
    """Aggregate complete folds and generate all comparison figures."""
    dataset_means = utils.load_complete_dataset_means(
        results_file=RESULTS_FILE,
        estimator_order=MODEL_ORDER,
        metrics=("accuracy", "fit_time"),
        n_splits=utils.N_SPLITS,
    )
    dataset_means.to_csv(DATASET_MEANS_FILE, index=False)

    utils.draw_significance_accuracy(
        dataset_results=dataset_means,
        estimator_order=MODEL_ORDER,
        output_file=OUTPUT_DIR / "critical_difference_accuracy.png",
        title="Accuracy — critical difference diagram",
        display_labels=DISPLAY_LABELS,
        alpha=ALPHA,
    )

    draw_pairwise_accuracy(dataset_means, OUTPUT_DIR)

    utils.draw_mean_fit_time(
        dataset_results=dataset_means,
        estimator_order=MODEL_ORDER,
        output_file=OUTPUT_DIR / "mean_fit_time.png",
        title="Distribution of mean training time",
        display_labels=DISPLAY_LABELS,
    )

    print(f"Comparison outputs saved to: {OUTPUT_DIR.resolve()}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare six interpretable classifiers on binary UCI data."
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip RFR compilation and estimator smoke tests.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate summaries and figures without fitting estimators.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help=(
            "Root directory for experiment outputs. "
            "Default: results."
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    paths = utils.make_output_paths(arguments.results_root, experiment_directory="comparison")

    OUTPUT_ROOT = paths["output_dir"]
    RESULTS_FILE = paths["results_file"]
    ERRORS_FILE = paths["errors_file"]
    DATASET_MEANS_FILE = paths["dataset_means_file"]
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not arguments.plots_only:
        if not arguments.skip_warmup:
            warm_up_estimators()

        utils.run_benchmark(
            datasets=COMPARISON_DATASETS,
            make_estimators=make_estimators,
            results_file=RESULTS_FILE,
            errors_file=ERRORS_FILE,
        )

    generate_outputs(OUTPUT_ROOT, RESULTS_FILE, DATASET_MEANS_FILE)


if __name__ == "__main__":
    main()
