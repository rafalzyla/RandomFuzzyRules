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


def clean_features(X, dataset):
    """Clean feature values and infer numerical/categorical columns."""
    X = X.copy()
    X = X.replace(MISSING_MARKERS, np.nan)

    variable_types = get_variable_type_map(dataset)

    categorical_columns = []
    numerical_columns = []
    columns_to_drop = []

    for column in X.columns:
        declared_type = variable_types.get(str(column), "").lower()

        # Remove a column if it contains no observed values.
        if X[column].notna().sum() == 0:
            columns_to_drop.append(column)
            continue

        # Prefer the variable type supplied by UCI.
        if declared_type in {"categorical", "binary"}:
            X[column] = X[column].astype("object")
            categorical_columns.append(column)
            continue

        if declared_type in {
            "integer",
            "continuous",
            "real",
            "numeric",
        }:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            ).astype(float)

            numerical_columns.append(column)
            continue

        # Fallback for variables whose type is missing or ambiguous.
        converted = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        original_non_missing = X[column].notna().sum()
        converted_non_missing = converted.notna().sum()

        conversion_ratio = (
            converted_non_missing / original_non_missing
        )

        if conversion_ratio >= 0.95:
            X[column] = converted.astype(float)
            numerical_columns.append(column)
        else:
            X[column] = X[column].astype("object")
            categorical_columns.append(column)

    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)

    return (
        X,
        numerical_columns,
        categorical_columns,
    )


def prepare_target(dataset_id, y_frame):
    """Convert a UCI target into a binary NumPy array."""
    if y_frame is None or y_frame.empty:
        raise ValueError("Dataset has no target column.")

    if y_frame.shape[1] != 1:
        raise ValueError(
            f"Expected one target column, found {y_frame.shape[1]}: "
            f"{list(y_frame.columns)}"
        )

    y = y_frame.iloc[:, 0].copy()
    y = y.replace(MISSING_MARKERS, np.nan)

    # Special case: UCI Heart Disease.
    # 0 = no disease; values 1-4 = presence of disease.
    if dataset_id == 45:
        y_numeric = pd.to_numeric(y, errors="coerce")
        y_binary = (y_numeric > 0).astype(float)
        y_binary[y_numeric.isna()] = np.nan
        return y_binary

    return y


def load_uci_dataset(dataset_id, expected_name=None):
    """Download and clean one UCI dataset."""

    dataset = fetch_ucirepo(id=dataset_id)

    X = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy()

    y = prepare_target(dataset_id, y_raw)

    # Remove observations with missing target.
    valid_target = pd.Series(y).notna().to_numpy()
    X = X.loc[valid_target].reset_index(drop=True)
    y = pd.Series(y).loc[valid_target].reset_index(drop=True)

    # Remove duplicated columns
    X = X.T.drop_duplicates(keep="first").T

    X, numerical_columns, categorical_columns = clean_features(X, dataset)

    # Generic binary encoding.
    unique_classes = pd.Series(y).dropna().unique()

    if len(unique_classes) != 2:
        raise ValueError(
            f"Dataset {dataset_id} is not binary after target preparation. "
            f"Classes: {unique_classes}"
        )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str)).astype(np.int64)

    result = {
        "dataset_id": dataset_id,
        "dataset_name": dataset.metadata.name,
        "X": X,
        "y": y_encoded,
        "target_classes": list(encoder.classes_),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }

    return result
    

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
            data = load_uci_dataset(
                dataset_id=dataset_id,
                expected_name=expected_name,
            )

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
