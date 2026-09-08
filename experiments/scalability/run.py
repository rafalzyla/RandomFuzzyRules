"""Single-thread scalability benchmarks for interpretable classifiers.

The study evaluates training-time scaling with respect to:

1. number of samples, for six classifiers;
2. number of original features, for six classifiers;
3. number of random candidates, for RandomFuzzyRules only.

Synthetic data are generated directly in their preprocessed representation.
Numerical columns lie in [0, 1], while each categorical feature is represented
by a valid four-column one-hot group. Data-generation time is excluded from
reported fitting times. Results are appended after every successful or failed
measurement, so interrupted studies can be resumed.

Run this module from the repository root, for example:

    uv run --project environments/main --locked \
        python -m experiments.scalability.run --study all
"""

from __future__ import annotations

# Thread-control variables must be set before importing NumPy, Numba,
# scikit-learn, SciPy, imodels, or any BLAS/OpenMP-dependent package.
import os

_SINGLE_THREAD_ENV = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _name, _value in _SINGLE_THREAD_ENV.items():
    os.environ[_name] = _value

import argparse
import gc
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from experiments import utils
from experiments.ablation.validate_default_configuration import SELECTED_DEFAULTS
from experiments.gpr_fast_bridge import GPRFastSubprocessClassifier
from random_fuzzy_rules import RandomFuzzyRulesClassifier


N_REPEATS = 5
BASE_N_SAMPLES = 1_000
BASE_N_NUMERICAL = 60
BASE_N_CATEGORICAL = 10
CATEGORIES_PER_FEATURE = 4

SAMPLE_VALUES = np.logspace(7, 17, 11, base=2).astype(int)
FEATURE_VALUES = np.logspace(2, 11, 10, base=2).astype(int)
CANDIDATE_VALUES = np.array([1_000, 5_000, 10_000, 25_000, 50_000, 100_000], dtype=int)

MODEL_ORDER = [
    "RFR",
    "GPR",
    "FIGS",
    "HSTree",
    "GreedyRuleList",
]

RESULT_COLUMNS = [
    "study",
    "value",
    "repeat",
    "estimator",
    "n_samples",
    "n_original_features",
    "n_numerical_features",
    "n_categorical_features",
    "n_transformed_features",
    "n_candidates",
    "fit_time",
    "fit_wall_time",
    "status",
]

N_JOBS = 1
numba.set_num_threads(N_JOBS)

# ---------------------------------------------------------------------------
# Synthetic preprocessed data
# ---------------------------------------------------------------------------
def _feature_split(n_original_features: int) -> tuple[int, int]:
    """Split original features so post-one-hot blocks are approximately equal.

    For four-category categorical variables, choosing approximately four
    numerical variables per categorical variable makes the numerical column
    count close to the one-hot column count after preprocessing.
    """
    if n_original_features < 1:
        raise ValueError("n_original_features must be positive.")
    if n_original_features == 1:
        return 1, 0

    n_categorical = max(1, int(round(n_original_features / (CATEGORIES_PER_FEATURE + 1))))
    n_categorical = min(n_categorical, n_original_features - 1)
    n_numerical = n_original_features - n_categorical
    return n_numerical, n_categorical


def make_preprocessed_classification_data(
    n_samples: int,
    n_numerical: int,
    n_categorical: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, ...]]]:
    """Generate dense membership data and a balanced nonlinear binary target."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two.")
    if n_numerical < 0 or n_categorical < 0:
        raise ValueError("Feature counts must be non-negative.")
    if n_numerical + n_categorical < 1:
        raise ValueError("At least one original feature is required.")

    rng = np.random.default_rng(random_state)
    n_transformed = n_numerical + CATEGORIES_PER_FEATURE * n_categorical
    X = np.empty((n_samples, n_transformed), dtype=np.float64)

    feature_names: list[str] = []
    if n_numerical:
        X[:, :n_numerical] = rng.random((n_samples, n_numerical))
        feature_names.extend(f"num_{j}" for j in range(n_numerical))

    categorical_groups: list[tuple[int, ...]] = []
    current = n_numerical
    # Generate one categorical feature at a time to avoid an additional large
    # integer matrix during the high-sample benchmark.
    for feature_index in range(n_categorical):
        categories = rng.integers(0, CATEGORIES_PER_FEATURE,size=n_samples)
        group = tuple(range(current, current + CATEGORIES_PER_FEATURE))
        X[:, current : current + CATEGORIES_PER_FEATURE] = 0.0
        X[:, current + categories] = 1.0
        categorical_groups.append(group)
        feature_names.extend(
            f"cat_{feature_index}={category}"
            for category in range(CATEGORIES_PER_FEATURE)
        )
        current += CATEGORIES_PER_FEATURE

    # Use only a bounded number of informative columns so that task difficulty
    # does not grow automatically with dimensionality. Median thresholding
    # yields a nearly balanced target for every generated dataset.
    signal = np.zeros(n_samples, dtype=np.float64)
    n_signal_numerical = min(n_numerical, 8)
    if n_signal_numerical:
        weights = np.linspace(1.0, 0.3, n_signal_numerical)
        signal += X[:, :n_signal_numerical] @ weights
        if n_signal_numerical >= 2:
            signal += 0.75 * X[:, 0] * X[:, 1]

    n_signal_categorical = min(n_categorical, 2)
    for group_index in range(n_signal_categorical):
        group_start = n_numerical + group_index * CATEGORIES_PER_FEATURE
        signal += 0.8 * X[:, group_start]
        signal -= 0.4 * X[:, group_start + 1]

    signal += rng.normal(0.0, 0.25, size=n_samples)
    y = (signal >= np.median(signal)).astype(np.int64)

    return (
        np.ascontiguousarray(X),
        np.ascontiguousarray(y),
        feature_names,
        categorical_groups,
    )


# ---------------------------------------------------------------------------
# Estimator construction
# ---------------------------------------------------------------------------
def _supported_kwargs(cls, **kwargs):
    """Retain only constructor arguments explicitly supported by a class."""
    parameters = inspect.signature(cls).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def make_estimator_factories(
    feature_names: list[str],
    continuous_features,
    categorical_feature_groups,
    rfr_candidates: int | None = None,
) -> dict[str, Callable[[], object]]:
    """Create fresh-estimator factories for the six benchmarked algorithms."""
    from imodels import (
        FIGSClassifier,
        GreedyRuleListClassifier,
        HSTreeClassifier
    )

    rfr_parameters = dict(SELECTED_DEFAULTS)
    if rfr_candidates is not None:
        rfr_parameters["n_candidates"] = int(rfr_candidates)
        # Preserve the selected configuration's two-attempts-per-target ratio.
        # A fixed 20,000 limit would make targets above 20,000 impossible.
        rfr_parameters["max_sampling_attempts"] = max(20_000, 2 * int(rfr_candidates))

    figs_kwargs = _supported_kwargs(
        FIGSClassifier,
        n_jobs=N_JOBS,
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
        "RFR": lambda: RandomFuzzyRulesClassifier(
            **rfr_parameters,
            continuous_features=continuous_features,
            categorical_feature_groups=categorical_feature_groups,
            feature_names=feature_names,
            random_state=utils.RANDOM_STATE,
        ),
        "GPR": lambda: GPRFastSubprocessClassifier(
            feature_names=feature_names,
            n_populations=100,
            n_generations=100,
            threshold=0.5,
            verbose=False,
            max_n_of_rules=6,
            max_n_of_ands=6,
            base_pb=0.1,
            random_state=utils.RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        "FIGS": lambda: FIGSClassifier(**figs_kwargs),
        "HSTree": lambda: HSTreeClassifier(**hstree_kwargs),
        "GreedyRuleList": lambda: GreedyRuleListClassifier(**greedy_kwargs),
    }


# ---------------------------------------------------------------------------
# Persistence and timing
# ---------------------------------------------------------------------------
def _load_results(RESULTS_FILE) -> pd.DataFrame:
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _append_row(path: Path, row: dict, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if columns is not None:
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _completed_keys(results: pd.DataFrame) -> set[tuple[str, int, int, str]]:
    if results.empty:
        return set()
    completed = results[results["status"] == "ok"]
    return {
        (str(row.study), int(row.value), int(row.repeat), str(row.estimator))
        for row in completed.itertuples()
    }


def _time_one_estimator(
    estimator_name: str,
    factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Fit one estimator and return algorithm time and observed wall time."""
    estimator = factory()
    wall_start = time.perf_counter()
    try:
        estimator.fit(X, y)
        wall_time = time.perf_counter() - wall_start
        fit_time = float(getattr(estimator, "_benchmark_fit_time_", wall_time))
        return fit_time, wall_time
    finally:
        if hasattr(estimator, "close"):
            estimator.close()
        del estimator
        gc.collect()


def _run_grid(
    RESULTS_FILE: Path, 
    ERRORS_FILE: Path, 
    study: str, 
    values: np.ndarray, 
    n_repeats: int, 
    max_data_gb: float | None,
) -> None:
    results = _load_results(RESULTS_FILE)
    completed = _completed_keys(results)

    for value in values:
        value = int(value)
        for repeat in range(n_repeats):
            random_state = utils.RANDOM_STATE + 100_000 * repeat + value

            if study == "samples":
                n_samples = value
                n_numerical = BASE_N_NUMERICAL
                n_categorical = BASE_N_CATEGORICAL
                candidate_count = int(SELECTED_DEFAULTS["n_candidates"])
                estimator_names = MODEL_ORDER
            elif study == "features":
                n_samples = BASE_N_SAMPLES
                n_numerical, n_categorical = _feature_split(value)
                candidate_count = int(SELECTED_DEFAULTS["n_candidates"])
                estimator_names = MODEL_ORDER
            elif study == "candidates":
                n_samples = BASE_N_SAMPLES
                n_numerical = BASE_N_NUMERICAL
                n_categorical = BASE_N_CATEGORICAL
                candidate_count = value
                estimator_names = ["RFR"]
            else:
                raise ValueError(f"Unknown study: {study!r}")

            n_transformed = n_numerical + CATEGORIES_PER_FEATURE * n_categorical
            estimated_gb = n_samples * n_transformed * 8 / (1024**3)
            if max_data_gb is not None and estimated_gb > max_data_gb:
                print(
                    f"SKIP {study}={value}, repeat={repeat}: X alone would use "
                    f"approximately {estimated_gb:.2f} GiB."
                )
                continue

            pending = [
                name
                for name in estimator_names
                if (study, value, repeat, name) not in completed
            ]
            if not pending:
                print(f"SKIP completed: {study}={value}, repeat={repeat}")
                continue

            print(
                f"\n{study}={value} | repeat={repeat + 1}/{n_repeats} | "
                f"samples={n_samples} | original_features={n_numerical + n_categorical} "
                f"| transformed_features={n_transformed}"
            )

            X, y, feature_names, categorical_groups = make_preprocessed_classification_data(
                n_samples=n_samples,
                n_numerical=n_numerical,
                n_categorical=n_categorical,
                random_state=random_state,
            )
            continuous_features = "all" if n_categorical == 0 else list(range(n_numerical))
            factories = make_estimator_factories(
                feature_names=feature_names,
                continuous_features=continuous_features,
                categorical_feature_groups=categorical_groups,
                rfr_candidates=candidate_count if study == "candidates" else None,
            )

            for estimator_name in pending:
                print(f"  {estimator_name:16s}", end="", flush=True)
                try:
                    fit_time, wall_time = _time_one_estimator(
                        estimator_name,
                        factories[estimator_name],
                        X,
                        y,
                    )
                    row = {
                        "study": study,
                        "value": value,
                        "repeat": repeat,
                        "estimator": estimator_name,
                        "n_samples": n_samples,
                        "n_original_features": n_numerical + n_categorical,
                        "n_numerical_features": n_numerical,
                        "n_categorical_features": n_categorical,
                        "n_transformed_features": n_transformed,
                        "n_candidates": candidate_count if estimator_name == "RFR" else np.nan,
                        "fit_time": fit_time,
                        "fit_wall_time": wall_time,
                        "status": "ok",
                    }
                    _append_row(RESULTS_FILE, row, RESULT_COLUMNS)
                    completed.add((study, value, repeat, estimator_name))
                    print(f" | fit={fit_time:.6f}s | wall={wall_time:.6f}s")
                except Exception as error:
                    print(f" | ERROR: {error}")
                    error_row = {
                        "study": study,
                        "value": value,
                        "repeat": repeat,
                        "estimator": estimator_name,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    _append_row(ERRORS_FILE, error_row)

            del X, y, feature_names, categorical_groups, factories
            gc.collect()


# ---------------------------------------------------------------------------
# Warm-up and plotting
# ---------------------------------------------------------------------------
def _warm_up() -> None:
    """Warm Numba and smoke-test every estimator on a tiny dataset."""
    X, y, feature_names, categorical_groups = make_preprocessed_classification_data(
        n_samples=64,
        n_numerical=8,
        n_categorical=2,
        random_state=utils.RANDOM_STATE,
    )
    factories = make_estimator_factories(
        feature_names=feature_names,
        continuous_features=list(range(8)),
        categorical_feature_groups=categorical_groups,
        rfr_candidates=50,
    )

    # Use a cheap GPR warm-up rather than the benchmark's 100x100 setting.
    factories["GPR"] = lambda: GPRFastSubprocessClassifier(
        feature_names=feature_names,
        n_populations=2,
        n_generations=2,
        threshold=0.5,
        verbose=False,
        max_n_of_rules=2,
        max_n_of_ands=2,
        base_pb=0.1,
        random_state=utils.RANDOM_STATE,
    )

    for name in MODEL_ORDER:
        print(f"Warm-up: {name}")
        estimator = factories[name]()
        try:
            estimator.fit(X, y)
            estimator.predict(X[:8])
        finally:
            if hasattr(estimator, "close"):
                estimator.close()
            del estimator
            gc.collect()
    print("Warm-up completed.")


def _format_seconds(seconds, _position=None):
    if not np.isfinite(seconds) or seconds < 0:
        return ""
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    parts = []
    if hours:
        parts.append(f"{hours} h")
    if minutes:
        parts.append(f"{minutes} min")
    if secs:
        parts.append(f"{secs} s")
    if milliseconds and not hours:
        parts.append(f"{milliseconds} ms")
    return " ".join(parts) if parts else "0 ms"


def _save_plot(results: pd.DataFrame, study: str, OUTPUT_ROOT: Path) -> None:
    subset = results[(results["study"] == study) & (results["status"] == "ok")].copy()
    if subset.empty:
        print(f"No successful results available for plot: {study}")
        return

    x_labels = {
        "samples": "Number of samples",
        "features": "Number of original features",
        "candidates": "Number of candidates",
    }
    title_labels = {
        "samples": "Training-time scalability with the number of samples",
        "features": "Training-time scalability with the number of features",
        "candidates": "RFR training-time scalability with the number of candidates",
    }

    sns.set_theme(style="whitegrid", font_scale=1.6)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=subset,
        x="value",
        y="fit_time",
        hue="estimator",
        style="estimator",
        markers=True,
        dashes=False,
        estimator="median",
        errorbar=("pi", 50),
        hue_order=[name for name in MODEL_ORDER if name in subset["estimator"].unique()],
        style_order=[name for name in MODEL_ORDER if name in subset["estimator"].unique()],
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ticks = sorted(subset["value"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(value):,}" for value in ticks], rotation=35, ha="center")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_seconds))
    ax.set_xlabel(x_labels[study])
    ax.set_ylabel("Median training time")
    ax.set_title(title_labels[study])
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.75))
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / f"{study}_fit_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_plots(RESULTS_FILE: Path, OUTPUT_ROOT: Path) -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file was not found: {RESULTS_FILE}")
    results = pd.read_csv(RESULTS_FILE)
    for study in ("samples", "features", "candidates"):
        _save_plot(results, study, OUTPUT_ROOT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=["all", "samples", "features", "candidates"],
        default="all",
        help="Scalability dimension to execute.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=N_REPEATS,
        help="Number of repeated fits per grid point and estimator.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip Numba compilation and estimator smoke tests.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from the existing timing CSV.",
    )
    parser.add_argument(
        "--max-data-gb",
        type=float,
        default=None,
        help=(
            "Optional safety limit for the size of the dense X array alone. "
            "Grid points exceeding the limit are skipped."
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


def main() -> None:
    args = parse_arguments()
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be positive.")
    if args.max_data_gb is not None and args.max_data_gb <= 0:
        raise ValueError("--max-data-gb must be positive.")

    paths = utils.make_output_paths(args.results_root, experiment_directory="scalability")

    OUTPUT_ROOT = paths["output_dir"]
    RESULTS_FILE = paths["results_file"]
    ERRORS_FILE = paths["errors_file"]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    numba.set_num_threads(1)

    if args.plots_only:
        generate_plots(RESULTS_FILE, OUTPUT_ROOT)
        return

    if not args.skip_warmup:
        _warm_up()

    studies = (
        ["samples", "features", "candidates"]
        if args.study == "all"
        else [args.study]
    )
    for study in studies:
        if study == "samples":
            values = SAMPLE_VALUES
        elif study == "features":
            values = FEATURE_VALUES
        else:
            values = CANDIDATE_VALUES
        _run_grid(
            RESULTS_FILE=RESULTS_FILE,
            ERRORS_FILE=ERRORS_FILE,
            study=study,
            values=values,
            n_repeats=args.n_repeats,
            max_data_gb=args.max_data_gb,
        )
        generate_plots(RESULTS_FILE, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
