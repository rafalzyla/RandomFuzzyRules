"""Validate the selected RandomFuzzyRules default configuration.

The experiment compares the original baseline configuration used during the
one-factor-at-a-time ablation study with the configuration selected from the
ablation results. Evaluation uses the same development datasets as the
ablation study and ten-fold stratified cross-validation provided by
``experiments.utils.run_benchmark``.

Outputs are stored under
``results/ablation/default_configuration_validation``:

- ``fold_results.csv``;
- ``errors.csv`` when errors occur;
- ``dataset_mean_results.csv``;
- ``pairwise_accuracy.png``;
- ``mean_fit_time.png``.

The benchmark supports resuming because completed dataset/fold/estimator
combinations are read from ``fold_results.csv`` and skipped.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.visualisation import plot_pairwise_scatter

from experiments import utils
from experiments.ablation.config import BASELINE_DEFAULTS, UCI_DATASETS
from random_fuzzy_rules import RandomFuzzyRulesClassifier


OUTPUT_DIR = Path("results") / "ablation" / "default_configuration_validation"
RESULTS_FILE = OUTPUT_DIR / "fold_results.csv"
ERRORS_FILE = OUTPUT_DIR / "errors.csv"

BASELINE_NAME = "RFR_BaselineDefaults"
SELECTED_NAME = "RFR_SelectedDefaults"
ESTIMATOR_ORDER = [BASELINE_NAME, SELECTED_NAME]
DISPLAY_LABELS = {
    BASELINE_NAME: "Baseline defaults",
    SELECTED_NAME: "Selected defaults",
}

# Candidate default configuration selected from the ablation results.
#
# This configuration must be validated against BASELINE_DEFAULTS before it
# is frozen for scalability and final classifier-comparison experiments.
SELECTED_DEFAULTS = {
    "max_rules": 6,
    "max_rules_len": 3,
    "max_literal_repetitions": 3,
    "threshold": 0.5,
    "n_candidates": 10_000,
    "max_sampling_attempts": 20_000,
    "sampling_type_number": 1,
    "sampling_type_length": 2,
    "preprocessed": True,
}

def make_estimators(
    continuous_features,
    categorical_feature_groups,
    transformed_feature_names,
):
    """Create the two RFR configurations compared in this experiment."""
    common = {
        "continuous_features": continuous_features,
        "categorical_feature_groups": categorical_feature_groups,
        "feature_names": transformed_feature_names,
        "random_state": utils.RANDOM_STATE,
    }

    return {
        BASELINE_NAME: RandomFuzzyRulesClassifier(
            **BASELINE_DEFAULTS,
            **common,
        ),
        SELECTED_NAME: RandomFuzzyRulesClassifier(
            **SELECTED_DEFAULTS,
            **common,
        ),
    }


def warm_up_numba():
    """Compile the Numba kernels excluded from recorded fit times."""
    rng = np.random.default_rng(utils.RANDOM_STATE)
    X_warm = rng.uniform(0.0, 1.0, size=(100, 4))
    y_warm = np.array([0, 1] * 50, dtype=np.int64)

    # Both configurations use the same compiled signatures. One small fit is
    # sufficient to compile generation, threshold-0.5 evaluation, and scoring.
    model = RandomFuzzyRulesClassifier(
        max_rules=2,
        max_rules_len=2,
        max_literal_repetitions=2,
        threshold=0.5,
        n_candidates=50,
        max_sampling_attempts=500,
        sampling_type_number=1,
        sampling_type_length=2,
        preprocessed=True,
        continuous_features="all",
        random_state=utils.RANDOM_STATE,
    )
    model.fit(X_warm, y_warm)
    model.predict(X_warm[:10])
    print("Numba warm-up completed.")


def draw_pairwise_accuracy(dataset_means):
    """Create a paired Accuracy scatter for baseline and selected defaults."""
    matrix = utils.metric_matrix(
        dataset_results=dataset_means,
        metric="accuracy",
        estimator_order=ESTIMATOR_ORDER,
        display_labels=None,
    )

    fig, ax = plot_pairwise_scatter(
        results_a=matrix[BASELINE_NAME].to_numpy(),
        results_b=matrix[SELECTED_NAME].to_numpy(),
        method_a=DISPLAY_LABELS[BASELINE_NAME],
        method_b=DISPLAY_LABELS[SELECTED_NAME],
        metric="accuracy",
        lower_better=False,
        statistic_tests=True,
        title="Baseline versus selected RFR defaults — Accuracy",
        figsize=(8, 8),
        best_on_top=False,
    )
    fig.savefig(OUTPUT_DIR / "pairwise_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return matrix


def generate_outputs():
    dataset_means = (
        utils.load_complete_dataset_means(
            results_file=RESULTS_FILE,
            estimator_order=ESTIMATOR_ORDER,
            metrics=("accuracy", "fit_time"),
        )
    )
    dataset_means.to_csv(OUTPUT_DIR / "dataset_mean_results.csv", index=False)

    draw_pairwise_accuracy(dataset_means)
    utils.draw_mean_fit_time(
        dataset_results=dataset_means,
        estimator_order=ESTIMATOR_ORDER,
        output_file=OUTPUT_DIR / "mean_fit_time.png",
        title="Distribution of mean training time — RFR defaults",
        display_labels=DISPLAY_LABELS,
    )
    print(f"Results and plots saved to: {OUTPUT_DIR.resolve()}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare baseline and selected RandomFuzzyRules defaults."
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the one-time Numba warm-up.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate summaries and plots without fitting estimators.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.plots_only:
        if not args.skip_warmup:
            warm_up_numba()

        utils.run_benchmark(
            dataset_dictionary=UCI_DATASETS,
            make_estimators=make_estimators,
            results_file=RESULTS_FILE,
            errors_file=ERRORS_FILE,
        )

    generate_outputs()


if __name__ == "__main__":
    main()
