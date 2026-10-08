"""Standalone equivalence checks for the optimized GPR evaluation path.

The script compares the original compiled-expression evaluation with the
Numba-accelerated encoded-rule evaluation for identical fitted chromosomes,
input matrices, thresholds, and labels.

A failed assertion or unexpected exception terminates the process with a
non-zero exit code. Successful completion prints a short summary.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np

from gpr_fast.classifier import (
    GPR_FAST,
    default_eval_function,
    evaluate_encoded_ruleset,
)


@dataclass(frozen=True)
class TestSummary:
    """Counts of completed equivalence checks."""

    fitness_cases: int
    fitted_models: int
    chromosomes: int
    prediction_vectors: int


def _original_default_eval_function(y_true, y_pred):
    """Original NumPy implementation retained for equivalence testing."""
    comparison = y_true == y_pred
    score = 0
    unique, counts = np.unique(comparison, return_counts=True)
    for predicted_correctly, count in zip(unique, counts):
        if predicted_correctly:
            score += count
        else:
            score -= 2 * count
    return score


def _assert_equal_scalar(actual, expected, context):
    """Raise an informative assertion for unequal scalar values."""
    if actual != expected:
        raise AssertionError(
            f"{context}: expected {expected!r}, obtained {actual!r}."
        )


def _test_default_eval_edge_cases():
    """Compare both fitness implementations on deterministic edge cases."""
    cases = [
        (np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)),
        (np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)),
        (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])),
        (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0])),
        (np.zeros(100, dtype=np.int64), np.zeros(100, dtype=np.int64)),
        (np.ones(100, dtype=np.int64), np.zeros(100, dtype=np.int64)),
        (
            np.array([0] * 99 + [1], dtype=np.int64),
            np.array([0] * 98 + [1, 1], dtype=np.int64),
        ),
    ]

    for case_index, (y_true, y_pred) in enumerate(cases):
        expected = _original_default_eval_function(y_true, y_pred)
        actual = default_eval_function(y_true, y_pred)

        _assert_equal_scalar(
            actual,
            expected,
            f"Fitness edge case {case_index}",
        )

    return len(cases)


def _test_default_eval_randomized(rng, n_trials):
    """Compare both fitness implementations on randomized binary arrays."""
    for trial in range(n_trials):
        n_samples = int(rng.integers(1, 10_001))
        y_true = rng.integers(0, 2, size=n_samples, dtype=np.int64)
        y_pred = rng.integers(0, 2, size=n_samples, dtype=np.int64)

        expected = _original_default_eval_function(y_true, y_pred)
        actual = default_eval_function(y_true, y_pred)

        _assert_equal_scalar(
            actual,
            expected,
            f"Randomized fitness trial {trial}",
        )

    return n_trials


def _make_binary_target(rng, n_samples):
    """Return randomized labels guaranteed to contain both classes."""
    y = np.arange(n_samples, dtype=np.int64) % 2
    rng.shuffle(y)
    return y


def _test_fitted_evaluation_equivalence(rng, n_trials, base_seed):
    """Compare original and optimized prediction paths on fitted chromosomes."""
    thresholds = (0.1, 0.3, 0.5, 0.7, 0.9)
    chromosome_count = 0
    prediction_vector_count = 0

    for trial in range(n_trials):
        trial_seed = base_seed + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)

        n_samples = int(rng.integers(24, 81))
        n_features = int(rng.integers(2, 9))
        threshold = thresholds[trial % len(thresholds)]

        # Include exact boundary values as well as values inside (0, 1).
        X = rng.random((n_samples, n_features), dtype=np.float64)
        if n_samples >= 5:
            X[:5, 0] = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = _make_binary_target(rng, n_samples)

        model = GPR_FAST(
            feature_names=[f"x{index + 1}" for index in range(n_features)],
            n_populations=8,
            n_generations=2,
            threshold=threshold,
            verbose=False,
            max_n_of_rules=3,
            max_n_of_ands=3,
            base_pb=0.1,
            random_state=trial_seed,
        )
        model.fit(X, y)

        chromosome = model._best_fit
        rule_columns, rule_lengths = model._chromosome_to_rule_columns(
            chromosome
        )

        original_function = model._compile_chromosome(chromosome)
        # Original implementation of GPR uses np.apply_along_axis
        original_scores = np.apply_along_axis(
            original_function,
            1,
            model.complemented_samples,
        )
        original_predictions = (
            original_scores > model.threshold
        ).astype(np.int32)

        optimized_predictions = evaluate_encoded_ruleset(
            model.complemented_samples,
            rule_columns,
            rule_lengths,
            model.threshold,
        )

        np.testing.assert_array_equal(
            optimized_predictions,
            original_predictions,
            err_msg=(
                f"Prediction mismatch in fitted-model trial {trial} "
                f"(seed={trial_seed}, threshold={threshold})."
            ),
        )

        original_fitness = _original_default_eval_function(
            model.sample_labels,
            original_predictions,
        )
        optimized_fitness = default_eval_function(
            model.sample_labels,
            optimized_predictions,
        )
        _assert_equal_scalar(
            optimized_fitness,
            original_fitness,
            f"Fitted-model fitness trial {trial}",
        )

        chromosome_count += 1
        prediction_vector_count += 1

    return chromosome_count, prediction_vector_count


def run_equivalence_checks(
    *,
    seed=42,
    fitness_trials=1000,
    fitted_model_trials=20,
):
    """Run all equivalence checks and return their execution counts."""
    if fitness_trials < 1:
        raise ValueError("fitness_trials must be positive.")
    if fitted_model_trials < 1:
        raise ValueError("fitted_model_trials must be positive.")

    rng = np.random.default_rng(seed)

    fitness_cases = _test_default_eval_edge_cases()
    fitness_cases += _test_default_eval_randomized(
        rng,
        fitness_trials,
    )
    chromosomes, prediction_vectors = (
        _test_fitted_evaluation_equivalence(
            rng,
            fitted_model_trials,
            seed,
        )
    )

    return TestSummary(
        fitness_cases=fitness_cases,
        fitted_models=fitted_model_trials,
        chromosomes=chromosomes,
        prediction_vectors=prediction_vectors,
    )


def parse_arguments():
    """Parse standalone test-script arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact equivalence of the original and Numba-accelerated "
            "GPR evaluation paths."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--fitness-trials",
        type=int,
        default=1000,
        help="Number of randomized fitness checks (default: 1000).",
    )
    parser.add_argument(
        "--fitted-model-trials",
        type=int,
        default=20,
        help="Number of randomized fitted-model checks (default: 20).",
    )
    return parser.parse_args()


def main():
    """Execute the checks and print a machine-readable success summary."""
    arguments = parse_arguments()
    summary = run_equivalence_checks(
        seed=arguments.seed,
        fitness_trials=arguments.fitness_trials,
        fitted_model_trials=arguments.fitted_model_trials,
    )

    print("GPR evaluation equivalence checks passed.")
    print(f"  Fitness cases: {summary.fitness_cases}")
    print(f"  Fitted models: {summary.fitted_models}")
    print(f"  Chromosomes: {summary.chromosomes}")
    print(f"  Prediction vectors: {summary.prediction_vectors}")


if __name__ == "__main__":
    main()
