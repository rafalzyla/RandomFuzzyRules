"""Configuration shared by RandomFuzzyRules ablation experiments."""

# Development datasets used for hyperparameter selection.
#
# These datasets must not be reused for the final comparison benchmark,
# because the selected RandomFuzzyRules configuration is informed by
# performance on this collection.
UCI_DATASETS = {
    14: "Breast Cancer",
    15: "Breast Cancer Wisconsin Original",
    17: "Breast Cancer Wisconsin Diagnostic",
    27: "Credit Approval",
    43: "Haberman Survival",
    45: "Heart Disease",
    46: "Hepatitis",
    52: "Ionosphere",
    74: "Musk Version 1",
    75: "Musk Version 2",
    94: "Spambase",
    95: "SPECT Heart",
    105: "Congressional Voting Records",
    144: "Statlog German Credit",
    151: "Connectionist Bench Sonar",
    161: "Mammographic Mass",
    174: "Parkinsons",
    176: "Blood Transfusion Service Center",
    222: "Bank Marketing",
    225: "Indian Liver Patient Dataset",
    264: "EEG Eye State",
    267: "Banknote Authentication",
    277: "Thoracic Surgery",
    327: "Phishing Websites",
    329: "Diabetic Retinopathy Debrecen",
    451: "Breast Cancer Coimbra",
    519: "Heart Failure Clinical Records",
    529: "Early Stage Diabetes Risk Prediction",
}


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
    "preprocessed": True,
}


# One-factor-at-a-time ablation grid. Every parameter not varied in a
# particular study retains its value from BASELINE_DEFAULTS.
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