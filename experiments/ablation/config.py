"""Configuration shared by RandomFuzzyRules ablation experiments."""

# Original reference configuration used during the one-factor-at-a-time
# ablation study.
BASELINE_DEFAULTS = {
    "max_rules": 6,
    "max_rules_len": 6,
    "max_literal_repetitions": 3,
    "threshold": 0.5,
    "n_candidates": 10_000,
    "max_sampling_attempts": 1_000_000,
    "sampling_type_number": 2,
    "sampling_type_length": 2,
    "quantile_transform": None,
    "preprocessed": True,
}


# One-factor-at-a-time ablation grid. Every parameter not varied in a
# particular study retains its value from BASELINE_DEFAULTS.
ABLATION_CONFIGS = {
    "quantile_transform": [
        None,
        "uniform",
        "normal"
    ],
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
    "threshold": [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ],
}