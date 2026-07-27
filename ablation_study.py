"""One-factor-at-a-time ablation study for RandomFuzzyRulesClassifier.

Each key in ABLATION_CONFIGS defines a separate study. For a selected study,
all parameters not varied retain their values from DEFAULTS. Results and errors
are written to separate CSV files under results/ablation_study/<study_name>/.
The same folder also receives:

- critical_difference_accuracy.png
- mean_fit_time.png
- dataset_mean_results.csv

The script supports resuming: run_benchmark skips estimator/fold combinations
already present in the study-specific fold_results.csv file.
"""

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


import utils
from random_fuzzy_rules_classifier import RandomFuzzyRulesClassifier
from ablation_config import (
    DEFAULTS,
    ABLATION_CONFIGS,
    OUTPUT_ROOT,
    _safe_name,
    _sampling_label,
    _value_label,
    _estimator_name,
    _study_paths
)


def _build_parameters(study_name: str, value: Any) -> dict[str, Any]:
    parameters = dict(DEFAULTS)

    if study_name == "sampling":
        parameters.update(value)
    else:
        parameters[study_name] = value

    return parameters


def make_ablation_estimators(study_name: str,) -> Callable[..., dict[str, RandomFuzzyRulesClassifier]]:
    """Create the estimator factory expected by utils.run_benchmark."""

    def factory(
        continuous_features,
        categorical_feature_groups,
        transformed_feature_names,
    ):
        estimators = {}

        for value in ABLATION_CONFIGS[study_name]:
            parameters = _build_parameters(study_name, value)
            name = _estimator_name(study_name, value)

            estimators[name] = RandomFuzzyRulesClassifier(
                **parameters,
                continuous_features=continuous_features,
                categorical_feature_groups=categorical_feature_groups,
                feature_names=transformed_feature_names,
                random_state=utils.RANDOM_STATE,
            )

        return estimators

    return factory


# ---------------------------------------------------------------------------
# Result preparation
# ---------------------------------------------------------------------------
def _configure_utils_output(study_name: str) -> Path:
    """Point utils persistence helpers at this study's own files."""
    study_dir, results_file, errors_file = _study_paths(study_name)
    utils.OUTPUT_DIR = study_dir
    utils.RESULTS_FILE = results_file
    utils.ERRORS_FILE = errors_file
    return study_dir


def run_ablation_study(study_name: str) -> None:
    if study_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation study: {study_name!r}")

    study_dir = _configure_utils_output(study_name)
    estimator_factory = make_ablation_estimators(study_name)

    print("\n" + "=" * 79)
    print(f"Ablation study: {study_name}")
    print(f"Output directory: {study_dir.resolve()}")
    print("Variants:")
    for value in ABLATION_CONFIGS[study_name]:
        print(f"  - {_value_label(study_name, value)}")
    print("=" * 79)

    utils.run_benchmark(
        dataset_dictionary=utils.UCI_DATASETS,
        make_estimators=estimator_factory,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RandomFuzzyRules one-factor-at-a-time ablations."
    )
    parser.add_argument(
        "--study",
        choices=["all", *ABLATION_CONFIGS.keys()],
        default="all",
        help="A single ablation key, or 'all'.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not run the one-time Numba warm-up.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    studies = (
        list(ABLATION_CONFIGS)
        if arguments.study == "all"
        else [arguments.study]
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not arguments.skip_warmup:
        utils.warm_up_numba()

    for study_name in studies:
        run_ablation_study(study_name)


if __name__ == "__main__":
    main()