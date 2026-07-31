"""One-factor-at-a-time ablation study for RandomFuzzyRulesClassifier.

Each key in ABLATION_CONFIGS defines a separate study. For a selected study,
all parameters not varied retain their values from DEFAULTS. Results and errors
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
from aeon.visualisation import plot_boxplot, plot_critical_difference
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from experiments import utils
from random_fuzzy_rules import RandomFuzzyRulesClassifier

ALPHA = 0.05

# ---------------------------------------------------------------------------
# Default RandomFuzzyRules configuration
# ---------------------------------------------------------------------------
DEFAULTS = {
    "max_rules": 6,
    "max_rules_len": 6,
    "max_literal_repetitions": 3,
    "threshold": 0.5,
    "n_candidates": 10_000,
    "max_sampling_attempts": 1_000_000,
    "sampling_type_number": 2,
    "sampling_type_length": 2,
    "preprocessed": True,
}

# ---------------------------------------------------------------------------
# One-factor-at-a-time ablation grid
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "max_rules": [2, 3, 4, 5, 6, 7],
    "max_rules_len": [2, 3, 4, 5, 6, 7],
    "max_literal_repetitions": [1, 2, 3],
    "n_candidates": [
        1_000,
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
    ],
    "max_sampling_attempts": [
        20_000,
        100_000,
        1_000_000,
    ],
    "sampling": [
        {
            "sampling_type_number": 1,
            "sampling_type_length": 1,
        },
        {
            "sampling_type_number": 1,
            "sampling_type_length": 2,
        },
        {
            "sampling_type_number": 2,
            "sampling_type_length": 1,
        },
        {
            "sampling_type_number": 2,
            "sampling_type_length": 2,
        },
    ],
}

OUTPUT_ROOT = Path("results") / "ablation"

UCI_DATASETS = {
    14: "Breast Cancer",
    15: "Breast Cancer Wisconsin Original",
    17: "Breast Cancer Wisconsin Diagnostic",
    27: "Credit Approval",
    43: "Haberman Survival",
    45: "Heart Disease",
    46: "Hepatitis",
    52: "Ionosphere",
    74: "Musk Version 1",
    75: "Musk Version 2",
    94: "Spambase",
    95: "SPECT Heart",
    105: "Congressional Voting Records",
    144: "Statlog German Credit",
    151: "Connectionist Bench Sonar",
    161: "Mammographic Mass",
    174: "Parkinsons",
    176: "Blood Transfusion Service Center",
    222: "Bank Marketing",
    225: "Indian Liver Patient Dataset",
    264: "EEG Eye State",
    267: "Banknote Authentication",
    277: "Thoracic Surgery",
    327: "Phishing Websites",
    329: "Diabetic Retinopathy Debrecen",
    451: "Breast Cancer Coimbra",
    519: "Heart Failure Clinical Records",
    529: "Early Stage Diabetes Risk Prediction",
}

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

def _study_paths(study_name: str) -> tuple[Path, Path, Path]:
    study_dir = OUTPUT_ROOT / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    return (
        study_dir,
        study_dir / "fold_results.csv",
        study_dir / "errors.csv",
    )

def _build_parameters(study_name: str, value: Any) -> dict[str, Any]:
    parameters = dict(DEFAULTS)

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


def run_ablation_study(study_name: str) -> None:
    if study_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation study: {study_name!r}")

    study_dir, results_file, errors_file = _study_paths(study_name)
    estimator_factory = make_ablation_estimators(study_name)

    print("\n" + "=" * 79)
    print(f"Ablation study: {study_name}")
    print(f"Output directory: {study_dir.resolve()}")
    print("Variants:")
    for value in ABLATION_CONFIGS[study_name]:
        print(f"  - {_value_label(study_name, value)}")
    print("=" * 79)

    utils.run_benchmark(
        dataset_dictionary=UCI_DATASETS,
        make_estimators=estimator_factory,
        results_file=results_file,
        errors_file=errors_file
    )

def _load_dataset_means(results_file: Path, study_name: str,) -> pd.DataFrame:
    if not results_file.exists():
        raise FileNotFoundError(f"Results file was not found: {results_file}")

    results = pd.read_csv(results_file)

    required_columns = {
        "dataset_id",
        "dataset_name",
        "estimator",
        "accuracy",
        "fit_time",
    }
    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {results_file}: {sorted(missing)}"
        )

    expected_estimators = [
        _estimator_name(study_name, value)
        for value in ABLATION_CONFIGS[study_name]
    ]

    results = results[results["estimator"].isin(expected_estimators)].copy()

    if results.empty:
        raise ValueError(f"No results found for study {study_name!r}.")

    dataset_means = (
        results.groupby(
            [
                "dataset_id",
                "dataset_name",
                "estimator",
            ],
            as_index=False,
        )
        .agg(
            accuracy=("accuracy", "mean"),
            fit_time=("fit_time", "mean"),
            completed_folds=("fold", "nunique"),
        )
    )
    
    dataset_means = dataset_means[dataset_means["completed_folds"] == utils.N_SPLITS].copy()
    
    complete_matrix = dataset_means.pivot(
        index=["dataset_id", "dataset_name"],
        columns="estimator",
        values="accuracy",
    )
    
    complete_datasets = complete_matrix.dropna(axis=0, how="any").index

    dataset_means = dataset_means.set_index(["dataset_id", "dataset_name"])
    dataset_means = dataset_means.loc[dataset_means.index.isin(complete_datasets)].reset_index()

    if dataset_means.empty:
        raise ValueError(
            f"No dataset is complete for every variant in {study_name!r}."
        )

    return dataset_means


def _display_labels(study_name: str) -> dict[str, str]:
    return {
        _estimator_name(study_name, value): _value_label(study_name, value)
        for value in ABLATION_CONFIGS[study_name]
    }


# ---------------------------------------------------------------------------
# Plot preparation
# ---------------------------------------------------------------------------
def metric_matrix(dataset_means: pd.DataFrame, metric: str, study_name: str) -> pd.DataFrame:
    """Return a complete dataset-by-variant matrix in configured order."""
    estimator_order = [
        _estimator_name(study_name, value)
        for value in ABLATION_CONFIGS[study_name]
    ]

    matrix = dataset_means.pivot(
        index="dataset_name",
        columns="estimator",
        values=metric,
    )

    available_columns = [
        estimator
        for estimator in estimator_order
        if estimator in matrix.columns
    ]

    matrix = matrix[available_columns]
    matrix = matrix.dropna(axis=0, how="any")

    if matrix.empty:
        raise ValueError(
            f"No complete datasets are available for metric {metric!r} "
            f"in study {study_name!r}."
        )

    if len(available_columns) != len(estimator_order):
        missing = [
            estimator
            for estimator in estimator_order
            if estimator not in available_columns
        ]
        raise ValueError(
            f"Missing ablation variants in study {study_name!r}: {missing}"
        )

    display_labels = _display_labels(study_name)
    matrix = matrix.rename(columns=display_labels)

    return matrix


def draw_critical_difference_accuracy(dataset_means: pd.DataFrame, study_name: str, output_file: Path) -> pd.DataFrame:
    """Create an aeon critical-difference diagram for mean Accuracy."""
    matrix = metric_matrix(
        dataset_means=dataset_means,
        metric="accuracy",
        study_name=study_name,
    )

    if matrix.shape[1] < 2:
        raise ValueError(
            "At least two ablation variants are required for a "
            "critical-difference diagram."
        )

    fig, ax = plot_critical_difference(
        scores=matrix.to_numpy(),
        labels=list(matrix.columns),
        lower_better=False,
        test="wilcoxon",
        correction="holm",
        alpha=ALPHA,
        width=max(8, 1.25 * matrix.shape[1]),
        textspace=2.0,
    )

    ax.set_title(
        f"Accuracy — critical difference diagram — {study_name}"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return matrix


def draw_mean_fit_time(dataset_means: pd.DataFrame, study_name: str, output_file: Path) -> pd.DataFrame:
    """Create an aeon boxplot of per-dataset mean training times."""
    matrix = metric_matrix(
        dataset_means=dataset_means,
        metric="fit_time",
        study_name=study_name,
    )

    fig, ax = plot_boxplot(
        results=matrix.to_numpy(),
        labels=list(matrix.columns),
        relative=False,
        plot_type="boxplot",
        outliers=True,
        title=f"Distribution of mean training time — {study_name}",
    )

    ax.set_ylabel("Mean training time per dataset (seconds)")
    ax.set_xlabel("Ablation variant")

    # Keep a linear y-axis. This is explicit so later changes to plotting
    # defaults do not silently switch the study to logarithmic scaling.
    ax.set_yscale("linear")

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return matrix


# ---------------------------------------------------------------------------
# Study execution
# ---------------------------------------------------------------------------
def generate_study_outputs(study_name: str) -> None:
    study_dir, results_file, _ = _study_paths(study_name)
    dataset_means = _load_dataset_means(results_file, study_name)

    dataset_means.to_csv(
        study_dir / "dataset_mean_results.csv",
        index=False,
    )

    draw_critical_difference_accuracy(
        dataset_means=dataset_means,
        study_name=study_name,
        output_file=study_dir / "critical_difference_accuracy.png",
    )

    draw_mean_fit_time(
        dataset_means=dataset_means,
        study_name=study_name,
        output_file=study_dir / "mean_fit_time.png",
    )

    print(f"Plots saved to: {study_dir.resolve()}")


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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not arguments.skip_warmup:
        warm_up_numba()

    for study_name in studies:
        run_ablation_study(study_name)
        generate_study_outputs(study_name)


if __name__ == "__main__":
    main()