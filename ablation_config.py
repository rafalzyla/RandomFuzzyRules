from pathlib import Path
from typing import Any
from re import sub

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

OUTPUT_ROOT = Path("results") / "ablation_study"

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