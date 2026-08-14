"""One-factor-at-a-time ablation study for RandomFuzzyRulesClassifier.

Each key in ABLATION_CONFIGS defines a separate study. For a selected study,
all parameters not varied retain their values from BASELINE_DEFAULTS. Results and errors
are written to separate CSV files under results/ablation/<study_name>/.
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
from re import sub

import numpy as np
import pandas as pd

from experiments import utils
from experiments.ablation.config import ABLATION_CONFIGS, BASELINE_DEFAULTS
from experiments.datasets import ABLATION_DATASETS
from random_fuzzy_rules import RandomFuzzyRulesClassifier

ALPHA = 0.05

# ---------------------------------------------------------------------------
# Configuration and naming helpers
# ---------------------------------------------------------------------------
def _safe_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _sampling_label(number_type: int, length_type: int) -> str:
    names = {1: "uniform", 2: "exponential"}
    return (
        f"number={names[number_type]}, "
        f"length={names[length_type]}"
    )


def _value_label(study_name: str, value: Any) -> str:
    if study_name == "sampling":
        return _sampling_label(
            value["sampling_type_number"],
            value["sampling_type_length"],
        )
    return f"{study_name}={value}"


def _estimator_name(study_name: str, value: Any) -> str:
    """Return a stable CSV identifier for one ablation variant."""
    if study_name == "sampling":
        return (
            "RandomFuzzyRules__sampling__"
            f"n{value['sampling_type_number']}_"
            f"l{value['sampling_type_length']}"
        )
    return f"RandomFuzzyRules__{study_name}__{_safe_name(value)}"

def _study_paths(output_root: Path, study_name: str) -> tuple[Path, Path, Path]:
    """Return output paths for one ablation study."""
    if study_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation study: {study_name!r}")

    study_dir = output_root / study_name

    study_dir.mkdir(parents=True,exist_ok=True,)

    return (
        study_dir,
        study_dir / "fold_results.csv",
        study_dir / "errors.csv",
    )

def _build_parameters(study_name: str, value: Any) -> dict[str, Any]:
    parameters = dict(BASELINE_DEFAULTS)

    if study_name == "sampling":
        parameters.update(value)
    else:
        parameters[study_name] = value

    return parameters


def make_ablation_estimators(study_name: str) -> Callable:
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


def run_ablation_study(OUTPUT_ROOT: Path, study_name: str) -> None:
    if study_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation study: {study_name!r}")

    study_dir, results_file, errors_file = _study_paths(OUTPUT_ROOT, study_name)
    estimator_factory = make_ablation_estimators(study_name)

    print("\n" + "=" * 79)
    print(f"Ablation study: {study_name}")
    print(f"Output directory: {study_dir.resolve()}")
    print("Variants:")
    for value in ABLATION_CONFIGS[study_name]:
        print(f"  - {_value_label(study_name, value)}")
    print("=" * 79)

    utils.run_benchmark(
        datasets=ABLATION_DATASETS,
        make_estimators=estimator_factory,
        results_file=results_file,
        errors_file=errors_file
    )


def _display_labels(study_name: str) -> dict[str, str]:
    return {
        _estimator_name(study_name, value): _value_label(study_name, value)
        for value in ABLATION_CONFIGS[study_name]
    }


# ---------------------------------------------------------------------------
# Study execution
# ---------------------------------------------------------------------------
def generate_study_outputs(OUTPUT_ROOT: Path, study_name: str) -> None:
    study_dir, results_file, _ = _study_paths(OUTPUT_ROOT, study_name)
    estimator_order = [
        _estimator_name(study_name, value)
        for value in ABLATION_CONFIGS[study_name]
    ]
    
    dataset_means = (
        utils.load_complete_dataset_means(
            results_file=results_file,
            estimator_order=estimator_order,
            metrics=("accuracy", "fit_time"),
        )
    )

    dataset_means.to_csv(
        study_dir / "dataset_mean_results.csv",
        index=False,
    )
    
    display_labels = _display_labels(study_name)
    
    utils.draw_significance_accuracy(
        dataset_results=dataset_means,
        estimator_order=estimator_order,
        output_file=study_dir / "critical_difference_accuracy.png",
        title=f"Accuracy — critical difference diagram — {study_name}",
        display_labels=display_labels,
        alpha=ALPHA,
    )
    
    utils.draw_mean_fit_time(
        dataset_results=dataset_means,
        estimator_order=estimator_order,
        output_file=study_dir / "mean_fit_time.png",
        title=f"Distribution of mean training time — {study_name}",
        display_labels=display_labels,
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
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Regenerate plots from existing ablation "
            "results without fitting estimators."
        ),
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


def warm_up_numba():
    rng = np.random.default_rng(utils.RANDOM_STATE)

    # Preprocessed data must lie in [0, 1].
    X_warm = rng.uniform(
        0.0,
        1.0,
        size=(100, 4),
    )

    y_warm = np.array([0, 1] * 50)

    warm_model = RandomFuzzyRulesClassifier(
        max_rules=2,
        max_rules_len=2,
        max_literal_repetitions=2,
        threshold=0.5,
        n_candidates=50,
        max_sampling_attempts=500,
        preprocessed=True,
        continuous_features="all",
        random_state=utils.RANDOM_STATE,
    )

    warm_model.fit(X_warm, y_warm)
    
    # Compile the prediction scoring function as well.
    warm_model.predict(X_warm[:10])
    print("Numba warm-up completed.")


def main() -> None:
    arguments = parse_arguments()

    studies = (
        list(ABLATION_CONFIGS)
        if arguments.study == "all"
        else [arguments.study]
    )

    output_root = arguments.results_root.expanduser().resolve() / "ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    if arguments.plots_only:
        for study_name in studies:
            generate_study_outputs(output_root, study_name)

        return

    if not arguments.skip_warmup:
        warm_up_numba()

    for study_name in studies:
        run_ablation_study(output_root, study_name)

        generate_study_outputs(output_root, study_name)


if __name__ == "__main__":
    main()