from aeon.visualisation import plot_boxplot, plot_critical_difference
from pathlib import Path
import pandas as pd
import argparse
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

ALPHA = 0.05

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
            ["dataset_id", "dataset_name", "estimator"],
            as_index=False,
        )
        .agg(
            accuracy=("accuracy", "mean"),
            fit_time=("fit_time", "mean"),
            completed_folds=("fold", "nunique"),
        )
    )

    # Plots compare only datasets completed for every ablation variant.
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
        title=f"Mean training time — {study_name}",
    )

    ax.set_ylabel("Mean training time (seconds)")
    ax.set_xlabel("Ablation variant")

    # Keep a linear y-axis. This is explicit so later changes to plotting
    # defaults do not silently switch the study to logarithmic scaling.
    ax.set_yscale("linear")

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

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
        description="Generate plots for ablation study."
    )
    parser.add_argument(
        "--study",
        choices=["all", *ABLATION_CONFIGS.keys()],
        default="all",
        help="A single ablation key, or 'all'.",
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

    for study_name in studies:
        generate_study_outputs(study_name)


if __name__ == "__main__":
    main()