import gc
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
import matplotlib.pyplot as plt
from aeon.visualisation import plot_boxplot, plot_significance
from data_foundry.collections import BEYOND_ARENA

RANDOM_STATE = 42

MISSING_MARKERS = [
    "?",
    "NA",
    "N/A",
    "na",
    "null",
    "NULL",
    "",
]


RESULT_COLUMNS = [
    "dataset_name",
    "repeat_id",
    "fold_id",
    "fold",
    "n_expected_folds",
    "estimator",
    "n_train",
    "n_test",
    "n_original_features",
    "n_transformed_features",
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "mcc",
    "fit_time",
    "predict_time",
]


def load_dataset(dataset_name):
    """Download and clean one BeyondArena dataset."""
    container = BEYOND_ARENA.get_dataset(name_or_uuid=dataset_name)

    splits = []
    
    for repeat_id, repeat_folds in container.experiment_metadata.splits.items():
        for fold_id, indices in repeat_folds.items():
            train_indices, test_indices = indices
    
            splits.append(
                {
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "train_indices": np.asarray(train_indices, dtype=np.int32),
                    "test_indices": np.asarray(test_indices, dtype=np.int32),
                }
            )

    X = container.dataset
    y = X.pop(container.task_metadata.target_column_name)
    
    if pd.Series(y).isna().any():
        raise ValueError(f"Dataset {dataset_name} contains missing target values.")
        
    n_classes = len(np.unique(y))

    if n_classes != 2:
        raise ValueError(f"Dataset {dataset_name} has {n_classes} classes.")

    X = X[X.columns.drop_duplicates(keep="first")]

    # filter out datatypes diffrent than numbers/objects/categories, e.g. datetimes
    numerical_columns = list(X.select_dtypes(include="number"))
    categorical_columns = list(X.select_dtypes(include=["category", "object", "bool"]))

    X = X[numerical_columns + categorical_columns] 

    if categorical_columns:
        X[categorical_columns] = X[categorical_columns].astype("object")

    X = X.replace(MISSING_MARKERS, np.nan)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(pd.Series(y).astype(str)).astype(np.int64)

    class_counts = np.bincount(y_encoded)
    
    if len(class_counts) != 2:
        raise ValueError(f"Dataset {dataset_name} produced {len(class_counts)} encoded classes.")

    return {
        "dataset_name": dataset_name,
        "X": X,
        "y": y_encoded,
        "target_classes": list(encoder.classes_),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "n_expected_folds": len(splits),
        "splits": splits,
    }


def get_transformed_feature_metadata(
    fitted_preprocessor,
    numerical_columns,
    categorical_columns,
    n_transformed_features
):
    """Describe continuous columns and one-hot feature groups.

    Returns
    -------
    metadata : dict
        Dictionary containing:

        continuous_features : list of int or {"all"}
            Indices of transformed numerical columns. If all transformed
            columns are numerical, the string ``"all"`` is returned.

        categorical_feature_groups : list of tuple of int
            Each tuple contains transformed-column indices derived from one
            original categorical feature.

    Raises
    ------
    RuntimeError
        If the reconstructed transformed-feature indices do not match the
        actual number of columns emitted by the fitted preprocessor.

    Notes
    -----
    When ``OneHotEncoder`` uses ``max_categories`` or ``min_frequency``,
    ``encoder.categories_`` still contains all categories observed during
    fitting. Categories marked as infrequent are represented by one shared
    output column. Consequently, the number of output columns for a feature
    may be smaller than ``len(encoder.categories_[i])``.
    """
    # ColumnTransformer emits numerical columns first because the numerical
    # transformer is added before the categorical transformer.
    n_numerical = len(numerical_columns)

    continuous_indices = list(range(n_numerical))

    categorical_groups = []

    if categorical_columns:
        categorical_pipeline = fitted_preprocessor.named_transformers_["categorical"]

        encoder = categorical_pipeline.named_steps["onehot"]

        current_index = n_numerical

        infrequent_categories = getattr(
            encoder,
            "infrequent_categories_",
            None,
        )

        for feature_index, categories in enumerate(encoder.categories_):
            if infrequent_categories is None:
                infrequent = None
            else:
                infrequent = infrequent_categories[feature_index]

            if infrequent is None:
                group_size = len(categories)
            else:
                # All categories listed in `infrequent` are replaced by one
                # shared `infrequent_sklearn` output column.
                group_size = len(categories) - len(infrequent) + 1

            group = tuple(range(current_index, current_index + group_size))

            categorical_groups.append(group)
            current_index += group_size

    if not categorical_groups:
        continuous_features = "all"
    else:
        continuous_features = continuous_indices

    described_indices = set(continuous_indices)

    for group in categorical_groups:
        described_indices.update(group)

    expected_indices = set(range(n_transformed_features))

    if described_indices != expected_indices:
        missing = sorted(
            expected_indices - described_indices
        )
        unexpected = sorted(
            described_indices - expected_indices
        )

        raise RuntimeError(
            "Inconsistent transformed-feature metadata. "
            f"Missing indices: {missing}; "
            f"unexpected indices: {unexpected}; "
            f"described columns: {len(described_indices)}; "
            f"actual transformed columns: "
            f"{n_transformed_features}."
        )

    return {
        "continuous_features": continuous_features,
        "categorical_feature_groups": categorical_groups
    }


def make_preprocessor(numerical_columns, categorical_columns):
    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "scaler",
                MinMaxScaler(clip=True),
            ),
        ])

        transformers.append(
            ("numeric", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "onehot",
                OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float64,
                max_categories=100,
            ),
            ),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable input columns.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def load_existing_results(results_file):
    results_file = Path(results_file)

    if results_file.exists():
        return pd.read_csv(results_file)

    return pd.DataFrame(columns=RESULT_COLUMNS)


def save_results(results, results_file):
    results.to_csv(results_file, index=False)


def append_error(error_record, errors_file):
    errors_file = Path(errors_file)
    
    error_df = pd.DataFrame([error_record])

    if errors_file.exists():
        old = pd.read_csv(errors_file)
        error_df = pd.concat([old, error_df], ignore_index=True)

    error_df.to_csv(errors_file, index=False)


def make_output_paths(results_root, experiment_directory):
    """Create standard output paths for one experiment."""
    output_dir = Path(results_root).expanduser().resolve() / Path(experiment_directory)

    return {
        "output_dir": output_dir,
        "results_file": output_dir / "fold_results.csv",
        "errors_file": output_dir / "errors.csv",
        "dataset_means_file": output_dir / "dataset_mean_results.csv"
    }


def get_positive_scores(estimator, X_test):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]

    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X_test)

    if hasattr(estimator, "predict"):
        return estimator.predict(X_test)

    raise AttributeError(
        "Estimator has neither predict_proba, decision_function, nor predict."
    )


def run_benchmark(datasets, make_estimators, results_file, errors_file):

    results_file = Path(results_file)
    errors_file = Path(errors_file)
    
    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    errors_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    results = load_existing_results(results_file)

    completed = {
        (
            str(row.dataset_name),
            int(row.repeat_id),
            int(row.fold_id),
            str(row.estimator),
        )
        for row in results.itertuples()
    }

    for dataset_position, dataset_name in enumerate(datasets, start=1):
        print(f"\n[{dataset_position}/{len(datasets)}] Loading: {dataset_name}")

        try:
            data = load_dataset(dataset_name=dataset_name)

            X = data["X"]
            y = np.asarray(data["y"], dtype=np.int64)

            class_counts = np.bincount(y)

            if len(class_counts) != 2:
                raise ValueError(f"Expected 2 classes, found {len(class_counts)}.")

            print(
                f"Dataset: {data['dataset_name']}; "
                f"samples={len(y)}; "
                f"features={X.shape[1]}; "
                f"class counts={class_counts.tolist()}"
            )

            preprocessor = make_preprocessor(
                numerical_columns=data["numerical_columns"],
                categorical_columns=data["categorical_columns"],
            )

            for fold, split in enumerate(data["splits"], start=1):
                repeat_id = split["repeat_id"]
                fold_id = split["fold_id"]
                
                X_train_raw = X.iloc[split["train_indices"]]
                X_test_raw = X.iloc[split["test_indices"]]
                
                y_train = y[split["train_indices"]]
                y_test = y[split["test_indices"]]
                
                # Fit preprocessing only on the training part of the fold.
                fold_preprocessor = clone(preprocessor)
                
                X_train = fold_preprocessor.fit_transform(X_train_raw)
                
                X_test = fold_preprocessor.transform(X_test_raw)

                # The fuzzy-rule classifiers and Numba kernels require dense,
                # contiguous floating-point arrays.
                X_train = np.ascontiguousarray(
                    X_train,
                    dtype=np.float64,
                )
                
                X_test = np.ascontiguousarray(
                    X_test,
                    dtype=np.float64,
                )

                feature_names = [f"x{i+1}" for i in range(X_train.shape[1])]
                
                # Ensure that preprocessing produced valid membership values.
                if not np.isfinite(X_train).all():
                    raise ValueError(
                        "X_train contains NaN or infinite values "
                        "after preprocessing."
                    )
                
                if not np.isfinite(X_test).all():
                    raise ValueError(
                        "X_test contains NaN or infinite values "
                        "after preprocessing."
                    )
                
                if (
                    np.any(X_train < 0.0)
                    or np.any(X_train > 1.0)
                    or np.any(X_test < 0.0)
                    or np.any(X_test > 1.0)
                ):
                    raise ValueError(
                        "Preprocessed values must lie in [0, 1]."
                    )
                
                feature_metadata = (
                    get_transformed_feature_metadata(
                        fitted_preprocessor=fold_preprocessor,
                        numerical_columns=data["numerical_columns"],
                        categorical_columns=data["categorical_columns"],
                        n_transformed_features=len(feature_names)
                    )
                )
                
                estimators = make_estimators(
                    continuous_features=feature_metadata["continuous_features"],
                    categorical_feature_groups=feature_metadata["categorical_feature_groups"],
                    transformed_feature_names=feature_names,
                )

                for estimator_name, estimator in estimators.items():
                    key = (
                        dataset_name,
                        repeat_id,
                        fold_id,
                        estimator_name,
                    )
                    
                    try:
                        if key in completed:
                            print(
                                f"  Fold {fold:02d} | {estimator_name}: "
                                f"already completed"
                            )
                            continue
    
                        print(
                            f"  Fold {fold:02d} | "
                            f"{estimator_name:22s}",
                            end="",
                            flush=True,
                        )

                        gc.collect()

                        fit_start = time.perf_counter()
                        estimator.fit(X_train, y_train)
                        fit_wall_time = time.perf_counter() - fit_start

                        fit_time = getattr(estimator, "_benchmark_fit_time_", fit_wall_time)

                        predict_start = time.perf_counter()
                        y_pred = estimator.predict(X_test)
                        predict_wall_time = time.perf_counter() - predict_start

                        y_score = get_positive_scores(estimator, X_test)

                        predict_time = getattr(estimator, "_benchmark_predict_time_", predict_wall_time)

                        record = {
                            "dataset_name": data["dataset_name"],
                            "repeat_id": repeat_id,
                            "fold_id": fold_id,
                            "fold": fold,
                            "n_expected_folds": data["n_expected_folds"],
                            "estimator": estimator_name,
                            "n_train": len(split["train_indices"]),
                            "n_test": len(split["test_indices"]),
                            "n_original_features": X.shape[1],
                            "n_transformed_features": X_train.shape[1],
                            "accuracy": accuracy_score(y_test, y_pred),
                            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                            "auroc": roc_auc_score(y_test, y_score),
                            "mcc": matthews_corrcoef(y_test, y_pred),
                            "fit_time": fit_time,
                            "predict_time": predict_time,
                        }

                        results = pd.concat(
                            [results, pd.DataFrame([record])],
                            ignore_index=True,
                        )

                        completed.add(key)

                        # Save after every classifier/fold pair.
                        save_results(results, results_file)

                        print(
                            f" | acc={record['accuracy']:.3f}"
                            f" | fit={fit_time:.3f}s"
                            f" | pred={predict_time:.6f}s"
                        )

                    except Exception as error:
                        print(f" | ERROR: {error}")

                        append_error({
                            "dataset_name": data['dataset_name'],
                            "repeat_id": repeat_id,
                            "fold_id": fold_id,
                            "fold": fold,
                            "n_expected_folds": data["n_expected_folds"],
                            "estimator": estimator_name,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "traceback": traceback.format_exc(),
                        }, errors_file)

                    finally:
                        if hasattr(estimator, "close"):
                            estimator.close()

        except Exception as error:
            print(f"Dataset ERROR: {error}")

            append_error({
                "dataset_name": dataset_name,
                "fold": None,
                "estimator": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }, errors_file)

    return results


def load_complete_dataset_means(
    results_file,
    estimator_order,
    metrics=("accuracy", "fit_time"),
):
    """Load and aggregate complete results by dataset.

    A dataset-estimator result is complete when the number of distinct
    evaluated splits equals the dataset-specific value stored in
    ``n_expected_folds``. Only datasets with complete results for every
    requested estimator are returned.
    """
    results_file = Path(results_file)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file was not found: {results_file}"
        )

    results = pd.read_csv(results_file)

    required_columns = {
        "dataset_name",
        "repeat_id",
        "fold_id",
        "n_expected_folds",
        "estimator",
        *metrics,
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {results_file}: "
            f"{sorted(missing_columns)}"
        )

    estimator_order = list(estimator_order)

    results = results[results["estimator"].isin(estimator_order)].copy()

    if results.empty:
        raise ValueError(f"No results were found for the requested estimators: {estimator_order}")

    results = results.drop_duplicates(
        subset=[
            "dataset_name",
            "repeat_id",
            "fold_id",
            "estimator",
        ],
        keep="last",
    )

    results["split_key"] = results["repeat_id"].astype(str) + "::" + results["fold_id"].astype(str)

    expected_folds_per_dataset = results.groupby("dataset_name")["n_expected_folds"].nunique()

    inconsistent_datasets = (
        expected_folds_per_dataset[
            expected_folds_per_dataset != 1
        ]
        .index
        .tolist()
    )

    if inconsistent_datasets:
        raise ValueError(f"Inconsistent n_expected_folds values for datasets: {inconsistent_datasets}")

    aggregation = {metric: (metric, "mean") for metric in metrics}

    aggregation.update({
        "completed_folds": (
            "split_key",
            "nunique",
        ),
        "n_expected_folds": (
            "n_expected_folds",
            "first",
        ),
    })

    dataset_means = (
        results.groupby(
            [
                "dataset_name",
                "estimator",
            ],
            as_index=False,
        )
        .agg(**aggregation)
    )

    dataset_means = dataset_means[
        dataset_means["completed_folds"]
        == dataset_means["n_expected_folds"]
    ].copy()

    if dataset_means.empty:
        raise ValueError("No estimator has a complete set of dataset-specific splits.")

    completeness_matrix = (
        dataset_means.pivot(
            index="dataset_name",
            columns="estimator",
            values="completed_folds",
        )
    )

    missing_estimators = [
        estimator
        for estimator in estimator_order
        if estimator
        not in completeness_matrix.columns
    ]

    if missing_estimators:
        raise ValueError(f"Missing complete results for estimators: {missing_estimators}")

    complete_dataset_index = (
        completeness_matrix[
            estimator_order
        ]
        .dropna(
            axis=0,
            how="any",
        )
        .index
    )

    dataset_means = dataset_means[
        dataset_means["dataset_name"].isin(complete_dataset_index)
    ].reset_index(drop=True)

    if dataset_means.empty:
        raise ValueError("No dataset has complete results for every requested estimator.")

    return dataset_means

# plotting utilities
def metric_matrix(
    dataset_results,
    metric,
    estimator_order,
    display_labels=None,
):
    """Build a complete dataset-by-estimator matrix for one metric.

    Parameters
    ----------
    dataset_results : pandas.DataFrame
        Dataset-level results containing ``dataset_name``, ``estimator``, and
        the selected metric. Values should already be averaged across folds.

    metric : str
        Name of the metric column to place in the matrix.

    estimator_order : sequence of str
        Estimator identifiers in the desired plotting order.

    display_labels : mapping of str to str or None, default=None
        Optional mapping from estimator identifiers to human-readable labels.

    Returns
    -------
    matrix : pandas.DataFrame
        Complete dataset-by-estimator matrix. Datasets with a missing result
        for any requested estimator are removed.
    """
    matrix = dataset_results.pivot(
        index="dataset_name",
        columns="estimator",
        values=metric,
    )

    missing_estimators = [
        estimator
        for estimator in estimator_order
        if estimator not in matrix.columns
    ]
    if missing_estimators:
        raise ValueError(
            "Missing estimators in dataset-level results: "
            f"{missing_estimators}"
        )

    matrix = matrix[list(estimator_order)]
    matrix = matrix.dropna(axis=0, how="any")

    if matrix.empty:
        raise ValueError(
            f"No complete datasets are available for metric {metric!r}."
        )

    if display_labels is not None:
        matrix = matrix.rename(columns=display_labels)

    return matrix


def draw_significance(
    dataset_results,
    estimator_order,
    output_file,
    title,
    metric="accuracy",
    display_labels=None,
    alpha=0.05,
):
    """Create and save an Accuracy critical-difference diagram."""
    matrix = metric_matrix(
        dataset_results=dataset_results,
        metric=metric,
        estimator_order=estimator_order,
        display_labels=display_labels,
    )

    if matrix.shape[1] < 2:
        raise ValueError(
            "At least two estimators are required for a critical-difference "
            "diagram."
        )

    fig, ax = plot_significance(
        scores=matrix.to_numpy(),
        labels=list(matrix.columns),
        lower_better=False,
        test="wilcoxon",
        correction="holm",
        alpha=alpha,
    )
    ax.set_title(title)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return matrix


def draw_mean_fit_time(
    dataset_results,
    estimator_order,
    output_file,
    title,
    display_labels=None,
):
    """Create and save a boxplot of per-dataset mean training times."""
    matrix = metric_matrix(
        dataset_results=dataset_results,
        metric="fit_time",
        estimator_order=estimator_order,
        display_labels=display_labels,
    )

    fig, ax = plot_boxplot(
        results=matrix.to_numpy(),
        labels=list(matrix.columns),
        relative=False,
        plot_type="boxplot",
        outliers=True,
        title=title,
    )
    ax.set_ylabel("Mean training time per dataset (seconds)")
    ax.set_xlabel("Configuration")
    ax.set_yscale("linear")

    ax.set_axisbelow(True)

    ax.grid(
        visible=True,
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return matrix