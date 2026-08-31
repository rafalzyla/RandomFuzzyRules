from __future__ import annotations

import argparse
import gc
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
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
    "HSTree",
    "GreedyRuleList",
    "Dummy",
]

DISPLAY_LABELS = {
    "RFR": "RFR",
    "GPR": "GPR",
    "FIGS": "FIGS",
    "HSTree": "HSTree",
    "GreedyRuleList": "GreedyRuleList",
    "Dummy": "Dummy",
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
    )
    from sklearn.dummy import DummyClassifier


    figs_kwargs = _supported_kwargs(
        FIGSClassifier,
        n_jobs=-1, # All threads are used
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
        "HSTree": HSTreeClassifier(**hstree_kwargs),
        "GreedyRuleList": GreedyRuleListClassifier(**greedy_kwargs),
        "Dummy": DummyClassifier(strategy="most_frequent", random_state=utils.RANDOM_STATE),
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


def draw_pairwise(dataset_means, metric, OUTPUT_DIR):
    """Compare RFR and GPR dataset-level mean mcc values."""
    matrix = utils.metric_matrix(
        dataset_results=dataset_means,
        metric=metric,
        estimator_order=["RFR", "GPR"],
        display_labels=None,
    )

    fig, _ = plot_pairwise_scatter(
        results_a=matrix["RFR"].to_numpy(),
        results_b=matrix["GPR"].to_numpy(),
        method_a="RFR",
        method_b="GPR",
        metric=metric,
        lower_better=False,
        statistic_tests=True,
        title=f"RFR versus GPR — {metric}",
        figsize=(8, 8),
        best_on_top=False,
    )
    output_file = OUTPUT_DIR / f"pairwise_rfr_vs_gpr_{metric}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return matrix


def draw_metric_radar(
    dataset_means,
    estimator_order,
    output_file,
    title,
    display_labels=None,
):
    """Create and save a radar chart of mean predictive performance.

    Accuracy, balanced Accuracy, and AUROC are displayed on their original
    ``[0, 1]`` scales. MCC is linearly mapped from ``[-1, 1]`` to ``[0, 1]``
    using ``(MCC + 1) / 2``.
    """
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "mcc",
    ]

    metric_labels = [
        "Accuracy",
        "Balanced\nAccuracy",
        "AUROC",
        "MCC\n(scaled)",
    ]

    estimator_order = list(estimator_order)

    metric_means = (
        dataset_means
        .groupby("estimator")[metrics]
        .mean()
        .reindex(estimator_order)
    )

    radar_values = metric_means.copy()

    radar_values["mcc"] = (radar_values["mcc"] + 1.0) / 2.0

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        len(metrics),
        endpoint=False,
    )

    closed_angles = np.concatenate([
        angles,
        angles[:1],
    ])

    colors = sns.color_palette("colorblind", n_colors=len(estimator_order))

    fig, ax = plt.subplots(
        figsize=(9, 8),
        subplot_kw={"projection": "polar"},
    )

    # Place Accuracy at the top and arrange the remaining metrics clockwise.
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels, fontsize=12)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        fontsize=11,
    )
    ax.set_rlabel_position(22.5)

    ax.grid(
        visible=True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
    )

    for estimator, color in zip(estimator_order, colors):
        values = radar_values.loc[
            estimator,
            metrics,
        ].to_numpy(dtype=np.float64)

        closed_values = np.concatenate([values, values[:1]])

        label = (
            display_labels.get(estimator, estimator)
            if display_labels is not None
            else estimator
        )

        ax.plot(
            closed_angles,
            closed_values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=label,
        )

        ax.fill(
            closed_angles,
            closed_values,
            color=color,
            alpha=0.08,
        )

    ax.set_title(title, fontsize=16, pad=24)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
        title="Classifier",
        frameon=True,
    )

    fig.text(
        0.5,
        0.02,
        "MCC is mapped from [-1, 1] to [0, 1] using (MCC + 1) / 2.",
        ha="center",
        fontsize=11,
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    plt.close(fig)

    return metric_means


def generate_outputs(OUTPUT_DIR, RESULTS_FILE, DATASET_MEANS_FILE):
    """Aggregate complete folds and generate all comparison figures."""
    dataset_means = utils.load_complete_dataset_means(
        results_file=RESULTS_FILE,
        estimator_order=MODEL_ORDER,
        metrics=(
            "accuracy", 
            "balanced_accuracy",
            "auroc",
            "mcc",
            "fit_time"
        ),
    )
    dataset_means.to_csv(DATASET_MEANS_FILE, index=False)

    for metric in ["accuracy", "balanced_accuracy", "auroc", "mcc"]:
        utils.draw_significance(
            dataset_results=dataset_means,
            estimator_order=MODEL_ORDER,
            metric=metric,
            output_file=OUTPUT_DIR / f"significance_{metric}.png",
            title=f"{metric} — significance diagram",
            display_labels=DISPLAY_LABELS,
            alpha=ALPHA,
        )

        draw_pairwise(dataset_means, metric, OUTPUT_DIR)

    draw_metric_radar(
        dataset_means=dataset_means,
        estimator_order=MODEL_ORDER,
        output_file=OUTPUT_DIR / "metric_radar.png",
        title="Mean predictive performance of classifiers",
        display_labels=DISPLAY_LABELS,
    )

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
