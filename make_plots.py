from aeon.visualisation import (
    plot_critical_difference,
    plot_boxplot,
)
from pandas import read_csv
from pathlib import Path

OUTPUT_DIR = Path("uci_benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "fold_results.csv"

def metric_matrix(dataset_results, metric):
    matrix = dataset_results.pivot(
        index="dataset_name",
        columns="estimator",
        values=metric,
    )

    available_columns = [
        estimator
        for estimator in ESTIMATOR_ORDER
        if estimator in matrix.columns
    ]

    matrix = matrix[available_columns]

    # Use only datasets completed by every classifier.
    matrix = matrix.dropna(axis=0, how="any")

    return matrix


def draw_critical_difference(
    dataset_results,
    metric,
    title,
    lower_better=False,
):
    matrix = metric_matrix(dataset_results, metric)

    fig, ax = plot_critical_difference(
        scores=matrix.to_numpy(),
        labels=list(matrix.columns),
        lower_better=lower_better,
        test="wilcoxon",
        correction="holm",
        alpha=0.05,
        width=8,
        textspace=2.0,
    )

    ax.set_title(title)

    output_file = OUTPUT_DIR / f"critical_difference_{metric}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return matrix

def draw_boxplot(
    dataset_results,
    metric,
    title,
    relative=False,
    log10=False,
):
    matrix = metric_matrix(dataset_results, metric)
    values = matrix.to_numpy()

    if log10:
        values = np.log10(
            np.maximum(values, np.finfo(float).tiny)
        )

    fig, ax = plot_boxplot(
        results=values,
        labels=list(matrix.columns),
        relative=relative,
        plot_type="boxplot",
        outliers=True,
        title=title,
    )

    output_file = OUTPUT_DIR / f"boxplot_{metric}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    if not RESULTS_FILE.exists():        
        raise FileNotFoundError(            
            f"Results file was not found: "            
            f"{RESULTS_FILE}"        
        )
    results = read_csv(RESULTS_FILE)
    
    dataset_results = (
        results
        .groupby(
            ["dataset_id", "dataset_name", "estimator"],
            as_index=False,
        )
        .agg(
            accuracy=("accuracy", "mean"),
            f1=("f1", "mean"),
            auroc=("auroc", "mean"),
            fit_time=("fit_time", "mean"),
            predict_time=("predict_time", "mean"),
        )
    )
    
    
    ESTIMATOR_ORDER = [
        "RandomFuzzyRules",
        "BeamFuzzyRules",
        "GPR",
        "DecisionTree",
        "LogisticRegression",
        "HistGradientBoosting"
    ]
    
    
    accuracy_matrix = draw_critical_difference(
        dataset_results,
        metric="accuracy",
        title="Accuracy — critical difference diagram",
        lower_better=False,
    )
    
    f1_matrix = draw_critical_difference(
        dataset_results,
        metric="f1",
        title="F1 — critical difference diagram",
        lower_better=False,
    )
    
    auroc_matrix = draw_critical_difference(
        dataset_results,
        metric="auroc",
        title="AUROC — critical difference diagram",
        lower_better=False,
    )
    
    draw_boxplot(
        dataset_results,
        metric="accuracy",
        title="Rozkład średniej Accuracy między zbiorami",
    )
    
    draw_boxplot(
        dataset_results,
        metric="f1",
        title="Rozkład średniej F1 między zbiorami",
    )
    
    draw_boxplot(
        dataset_results,
        metric="auroc",
        title="Rozkład średniej AUROC między zbiorami",
    )
    
    draw_boxplot(
        dataset_results,
        metric="fit_time",
        title="Średni czas uczenia",
        log10=False,
    )
    
    draw_boxplot(
        dataset_results,
        metric="predict_time",
        title="Średni czas predykcji",
        log10=False,
    )