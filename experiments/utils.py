import gc
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from aeon.visualisation import plot_boxplot, plot_critical_difference

RANDOM_STATE = 42
N_SPLITS = 10

MISSING_MARKERS = [
    "?",
    "NA",
    "N/A",
    "na",
    "null",
    "NULL",
    "",
]


def get_variable_type_map(dataset):
    """Return a mapping: variable name -> UCI variable type."""
    variables = dataset.variables

    if variables is None or variables.empty:
        return {}

    result = {}

    for _, row in variables.iterrows():
        name = row.get("name")
        role = str(row.get("role", "")).lower()
        variable_type = str(row.get("type", ""))

        if role == "feature" and name is not None:
            result[str(name)] = variable_type

    return result

DROP_COLUMNS = {
    # DARWIN: participant identifier.
    732: {"ID",},
}

CATEGORICAL_COLUMNS = {
    # Default of Credit Card Clients.
    350: {
        "X2",
        "X3",
        "X4",
    },

    # Online Shoppers Purchasing Intention.
    468: {
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    },
}

def normalize_text_series(series):
    """Normalize textual UCI values while preserving missing entries."""
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(
            series.dtype,
            pd.CategoricalDtype,
        )
    ):
        return series

    normalized = series.astype("string").str.strip()

    missing_values = {marker.lower() for marker in MISSING_MARKERS if marker}
    missing_mask = normalized.str.lower().isin(missing_values)

    normalized = normalized.mask(missing_mask, pd.NA)

    # Return object dtype with ordinary np.nan values. This interacts more
    # predictably with SimpleImputer and OneHotEncoder than pd.NA in a mixed
    # object column.
    normalized = normalized.astype("object").where(normalized.notna(), np.nan)

    return normalized

def resolve_columns(available_columns, requested_columns):
    """Resolve configured column names case-insensitively."""
    available_by_normalized_name = {
        str(column).strip().casefold(): column
        for column in available_columns
    }

    resolved = set()

    for requested in requested_columns:
        key = str(requested).strip().casefold()

        if key in available_by_normalized_name:
            resolved.add(available_by_normalized_name[key])

    return resolved

def clean_features(X, dataset, dataset_id):
    """Clean feature values and infer numerical/categorical columns."""
    X = X.copy()

    # Normalize whitespace and textual missing-value markers before type
    # inference. This is especially important for Adult and Chronic Kidney
    # Disease.
    for column in X.columns:
        X[column] = normalize_text_series(X[column])

    X = X.replace(MISSING_MARKERS, np.nan)

    requested_drop_columns = DROP_COLUMNS.get(dataset_id, set())

    resolved_drop_columns = (
        resolve_columns(
            available_columns=X.columns,
            requested_columns=requested_drop_columns,
        )
    )

    if resolved_drop_columns:
        X = X.drop(columns=list(resolved_drop_columns))

    requested_categorical = CATEGORICAL_COLUMNS.get(dataset_id, set())

    forced_categorical = (
        resolve_columns(
            available_columns=X.columns,
            requested_columns=requested_categorical,
        )
    )

    variable_types = get_variable_type_map(dataset)

    categorical_columns = []
    numerical_columns = []
    columns_to_drop = []

    for column in X.columns:
        declared_type = variable_types.get(str(column), "").strip().lower()

        if X[column].notna().sum() == 0:
            columns_to_drop.append(column)
            continue

        if column in forced_categorical or declared_type in {"categorical", "binary"}:
            X[column] = normalize_text_series(X[column]).astype("object")

            categorical_columns.append(column)
            continue

        if declared_type in {"integer", "continuous", "real", "numeric"}:
            converted = pd.to_numeric(X[column], errors="coerce").astype(float)
            if converted.notna().sum() == 0:
                columns_to_drop.append(column)
                continue
            
            X[column] = converted
            numerical_columns.append(column)
            continue

        converted = pd.to_numeric(X[column], errors="coerce",)

        original_non_missing = X[column].notna().sum()

        converted_non_missing = converted.notna().sum()

        if original_non_missing == 0:
            columns_to_drop.append(column)
            continue

        conversion_ratio = converted_non_missing / original_non_missing

        if conversion_ratio >= 0.95:
            X[column] = (converted.astype(float))
            numerical_columns.append(column)
        else:
            X[column] = normalize_text_series(X[column]).astype("object")
            categorical_columns.append(column)

    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)

        categorical_columns = [
            column
            for column in categorical_columns
            if column not in columns_to_drop
        ]

        numerical_columns = [
            column
            for column in numerical_columns
            if column not in columns_to_drop
        ]

    if not numerical_columns and not categorical_columns:
        raise ValueError(f"Dataset {dataset_id} has no usable feature columns.")

    return (
        X,
        numerical_columns,
        categorical_columns,
    )

def prepare_target(dataset_id, y_frame):
    """Convert a UCI target into a binary target"""
    if y_frame is None or y_frame.empty:
        raise ValueError("Dataset has no target column.")

    if y_frame.shape[1] != 1:
        raise ValueError(
            f"Expected one target column for dataset "
            f"{dataset_id}, found {y_frame.shape[1]}: "
            f"{list(y_frame.columns)}"
        )

    y = y_frame.iloc[:, 0].copy()

    y = normalize_text_series(y)

    # Adult sometimes contains labels originating from separate train and
    # test files, where test labels may have a trailing period.
    if dataset_id == 2:
        y = (
            pd.Series(y)
            .astype("string")
            .str.strip()
            .str.removesuffix(".")
            .astype("object")
        )

        y = y.where(pd.Series(y).notna(), np.nan)

    # UCI Heart Disease.
    # 0 = no disease; values 1-4 = presence of disease.
    if dataset_id == 45:
        y_numeric = pd.to_numeric(y, errors="coerce")
        y_binary = (y_numeric > 0).astype(float)
        y_binary[y_numeric.isna()] = np.nan
        return y_binary

    # Vertebral Column officially defines both a three-class and a binary
    # task. Disk Hernia and Spondylolisthesis are merged into Abnormal.
    if dataset_id == 212:
        normalized = (
            pd.Series(y)
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        mapping = {
            "no": "Normal",
            "normal": "Normal",
            "dh": "Abnormal",
            "sl": "Abnormal",
            "ab": "Abnormal",
            "abnormal": "Abnormal",
        }

        mapped = normalized.map(mapping)

        mapped[normalized.isna()] = np.nan

        unknown_values = sorted(set(normalized.dropna().unique()) - set(mapping))

        if unknown_values:
            raise ValueError(
                "Unexpected Vertebral Column target values: "
                f"{unknown_values}"
            )

        y = mapped

    y = pd.Series(y).replace(MISSING_MARKERS, np.nan)

    return y

def load_uci_dataset(dataset_id):
    """Download and clean one UCI dataset."""
    dataset = fetch_ucirepo(id=dataset_id)

    if dataset.data.features is None or dataset.data.features.empty:
        raise ValueError(f"Dataset {dataset_id} has no feature matrix.")

    if dataset.data.targets is None or dataset.data.targets.empty:
        raise ValueError(f"Dataset {dataset_id} has no target matrix.")

    X = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy()

    y = prepare_target(dataset_id=dataset_id, y_frame=y_raw)

    valid_target = pd.Series(y).notna().to_numpy()

    X = X.loc[valid_target].reset_index(drop=True)

    y = pd.Series(y).loc[valid_target].reset_index(drop=True)

    X = X.T.drop_duplicates(keep="first").T

    (
        X,
        numerical_columns,
        categorical_columns,
    ) = clean_features(
        X=X,
        dataset=dataset,
        dataset_id=dataset_id,
    )

    # Normalize target labels once more after filtering. This ensures that
    # labels differing only in whitespace are treated as one class.
    y = normalize_text_series(y)

    if pd.Series(y).isna().any():
        raise RuntimeError(
            f"Dataset {dataset_id} still contains missing target values after filtering."
        )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(pd.Series(y).astype(str)).astype(np.int64)

    class_counts = np.bincount(y_encoded)
    
    if len(class_counts) != 2:
        raise ValueError(f"Dataset {dataset_id} produced {len(class_counts)} encoded classes.")
    
    if class_counts.min() < N_SPLITS:
        raise ValueError(
            f"Dataset {dataset_id} has only "
            f"{class_counts.min()} observations in the "
            f"smallest class, fewer than "
            f"n_splits={N_SPLITS}."
        )

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.metadata.name,
        "X": X,
        "y": y_encoded,
        "target_classes": list(encoder.classes_),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
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
    continuous_features : list of int or str
        Indices of transformed numerical columns. If every output
        column is numerical, the string "all" is returned.

    categorical_feature_groups : list of tuple of int
        Each tuple contains transformed-column indices belonging
        to one original categorical variable.
    """
    # ColumnTransformer emits transformers in their declared order.
    # In make_preprocessor, numeric columns are added first.
    n_numerical = len(numerical_columns)

    continuous_indices = list(range(n_numerical))

    categorical_groups = []

    if categorical_columns:
        categorical_pipeline = (
            fitted_preprocessor.named_transformers_[
                "categorical"
            ]
        )

        encoder = (
            categorical_pipeline.named_steps["onehot"]
        )

        current_index = n_numerical

        for categories in encoder.categories_:
            group_size = len(categories)

            group = tuple(
                range(
                    current_index,
                    current_index + group_size,
                )
            )

            categorical_groups.append(group)
            current_index += group_size

    if not categorical_groups:
        # Convenient shortcut required by the classifier API.
        continuous_features = "all"
    else:
        continuous_features = continuous_indices

    described_indices = set(continuous_indices)

    for group in categorical_groups:
        described_indices.update(group)

    expected_indices = set(
        range(n_transformed_features)
    )

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

RESULT_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "fold",
    "estimator",
    "n_train",
    "n_test",
    "n_original_features",
    "n_transformed_features",
    "accuracy",
    "f1",
    "auroc",
    "fit_time",
    "predict_time",
]


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


def run_benchmark(dataset_dictionary, make_estimators, results_file, errors_file):

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
            int(row.dataset_id),
            int(row.fold),
            str(row.estimator),
        )
        for row in results.itertuples()
    }

    for dataset_position, (dataset_id, expected_name) in enumerate(
        dataset_dictionary.items(),
        start=1,
    ):
        print(
            f"\n[{dataset_position}/{len(dataset_dictionary)}] "
            f"Loading {dataset_id}: {expected_name}"
        )

        try:
            data = load_uci_dataset(dataset_id=dataset_id)

            X = data["X"]
            y = np.asarray(data["y"], dtype=np.int64)

            class_counts = np.bincount(y)

            if len(class_counts) != 2:
                raise ValueError(
                    f"Expected 2 classes, found {len(class_counts)}."
                )

            if class_counts.min() < N_SPLITS:
                raise ValueError(
                    f"The smallest class has only {class_counts.min()} "
                    f"instances, fewer than n_splits={N_SPLITS}."
                )

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

            cv = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE,
            )

            for fold, (train_indices, test_indices) in enumerate(
                cv.split(X, y),
                start=1,
            ):
                X_train_raw = X.iloc[train_indices]
                X_test_raw = X.iloc[test_indices]
                
                y_train = y[train_indices]
                y_test = y[test_indices]
                
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
                    key = (dataset_id, fold, estimator_name)

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

                    try:
                        gc.collect()

                        fit_start = time.perf_counter()
                        estimator.fit(X_train, y_train)
                        fit_time = time.perf_counter() - fit_start

                        predict_start = time.perf_counter()
                        y_pred = estimator.predict(X_test)
                        predict_time = time.perf_counter() - predict_start

                        y_score = get_positive_scores(
                            estimator,
                            X_test,
                        )

                        record = {
                            "dataset_id": dataset_id,
                            "dataset_name": data["dataset_name"],
                            "fold": fold,
                            "estimator": estimator_name,
                            "n_train": len(train_indices),
                            "n_test": len(test_indices),
                            "n_original_features": X.shape[1],
                            "n_transformed_features": X_train.shape[1],
                            "accuracy": accuracy_score(y_test, y_pred),
                            "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
                            "auroc": roc_auc_score(y_test, y_score),
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
                            f" | f1={record['f1']:.3f}"
                            f" | auc={record['auroc']:.3f}"
                            f" | fit={fit_time:.3f}s"
                            f" | pred={predict_time:.6f}s"
                        )

                    except Exception as error:
                        print(f" | ERROR: {error}")

                        append_error({
                            "dataset_id": dataset_id,
                            "dataset_name": expected_name,
                            "fold": fold,
                            "estimator": estimator_name,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "traceback": traceback.format_exc(),
                        }, errors_file)

        except Exception as error:
            print(f"Dataset ERROR: {error}")

            append_error({
                "dataset_id": dataset_id,
                "dataset_name": expected_name,
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
    n_splits=N_SPLITS,
):
    """Load and aggregate complete cross-validation results by dataset.

    Fold-level results are averaged separately for every dataset and
    estimator. Only dataset-estimator combinations containing exactly
    ``n_splits`` distinct folds are retained. Finally, only datasets with
    complete results for every requested estimator are returned.

    Parameters
    ----------
    results_file : str or pathlib.Path
        Path to the fold-level CSV file.

    estimator_order : sequence of str
        Identifiers of estimators required for the comparison. The order is
        preserved by downstream plotting functions.

    metrics : sequence of str, default=("accuracy", "fit_time")
        Fold-level metric columns to average for every dataset-estimator
        combination.

    n_splits : int, default=N_SPLITS
        Required number of distinct folds for a result to be considered
        complete.

    Returns
    -------
    dataset_means : pandas.DataFrame
        Dataset-level results containing one row per complete
        dataset-estimator combination. The returned frame includes averaged
        metric columns and ``completed_folds``.

    Raises
    ------
    FileNotFoundError
        If ``results_file`` does not exist.

    ValueError
        If required columns are absent, requested estimators have no results,
        or no dataset contains complete results for all estimators.
    """
    results_file = Path(results_file)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file was not found: {results_file}"
        )

    results = pd.read_csv(results_file)

    required_columns = {
        "dataset_id",
        "dataset_name",
        "fold",
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
        raise ValueError(
            "No results were found for the requested "
            f"estimators: {estimator_order}"
        )

    aggregation = {
        metric: (metric, "mean")
        for metric in metrics
    }

    aggregation["completed_folds"] = ("fold", "nunique")

    dataset_means = (
        results.groupby(
            [
                "dataset_id",
                "dataset_name",
                "estimator",
            ],
            as_index=False,
        )
        .agg(**aggregation)
    )

    dataset_means = dataset_means[dataset_means["completed_folds"] == n_splits].copy()

    if dataset_means.empty:
        raise ValueError(
            "No estimator has a complete set of "
            f"{n_splits} folds."
        )

    completeness_matrix = (
        dataset_means.pivot(
            index=[
                "dataset_id",
                "dataset_name",
            ],
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
        raise ValueError(
            "Missing complete results for estimators: "
            f"{missing_estimators}"
        )

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

    dataset_means = dataset_means.set_index(["dataset_id", "dataset_name"])

    dataset_means = dataset_means.loc[
        dataset_means.index.isin(
            complete_dataset_index
        )
    ].reset_index()

    if dataset_means.empty:
        raise ValueError(
            "No dataset has complete results for every requested estimator."
        )

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


def draw_critical_difference_accuracy(
    dataset_results,
    estimator_order,
    output_file,
    title,
    display_labels=None,
    alpha=0.05,
):
    """Create and save an Accuracy critical-difference diagram."""
    matrix = metric_matrix(
        dataset_results=dataset_results,
        metric="accuracy",
        estimator_order=estimator_order,
        display_labels=display_labels,
    )

    if matrix.shape[1] < 2:
        raise ValueError(
            "At least two estimators are required for a critical-difference "
            "diagram."
        )

    fig, ax = plot_critical_difference(
        scores=matrix.to_numpy(),
        labels=list(matrix.columns),
        lower_better=False,
        test="wilcoxon",
        correction="holm",
        alpha=alpha,
        width=max(8, 1.25 * matrix.shape[1]),
        textspace=2.0,
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

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return matrix