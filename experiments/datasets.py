"""Dataset partitions used in the experiments.

The ablation partition contains all datasets previously used during RFR
development (ablation), supplemented with 15 datasets sampled without 
replacement from the remaining benchmark pool using NumPy's default 
random generator and seed 42.

The resulting dataset lists are stored explicitly below and are not sampled
again during experiment execution.
"""

from numpy.random import default_rng


DATASET_SPLIT_RANDOM_STATE = 42


ALL_DATASETS = [
    'amazon_employee_access',
    'aps_failure',
    'bad_customer_detection',
    'bank_customer_churn',
    'bank_marketing',
    'bioresponse',
    'blood_transfusion',
    'churn',
    'coil_2000',
    'credit_card_clients_default',
    'credit_g',
    'customer_satisfaction_in_airline',
    'diabetes_130_us',
    'ecommerce_shipping',
    'fitness_club',
    'give_me_some_credit',
    'hazelnut_spread_contaminant_detection',
    'heloc',
    'hiva_agnostic',
    'hr_analytics',
    'in_vehicle_coupon_recommendation',
    'jm1',
    'kdd_cup_09_appetency',
    'marketing_campaign',
    'polish_companies_bankruptcy',
    'qsar_biodeg',
    'seismic_bumps',
    'taiwanese_bankruptcy_prediction',
    'credit_approval',
    'drug_induced_autoimmunity_prediction',
    'early_stage_diabetes_risk_prediction',
    'gallstone_disease',
    'heart_disease_cleveland',
    'heart_disease_hungary',
    'heart_disease_va_long_beach',
    'heart_failure_followup_survival',
    'hepatitis_survival_prediction',
    'home_credit_default_risk',
    'homesite_quote_conversion',
    'indian_liver_patient_dataset',
    'labour_inspection_compliance',
    'ljubljana_breast_cancer',
    'porto_seguro',
    'pva_revenue_prediction_kddcup98',
    'regensburg_pediatric_appendicitis',
    'santander_customer_satisfaction',
    'south_africa_coronary_heart_disease',
    'thyroid_discordant',
    'tour_travels_churn',
    'wids_diabetes_mellitus',
    'iranian_churn',
    'homeq_default_prediction',
    'clock_protein_toxicity',
    'prostate_cancer_detection',
    'lung_cancer_epithelial_genexp',
    'acquire_valued_shoppers_challenge',
    'home_credit_default_stability',
    'kick',
    'hotel_booking_demand',
    'ieee_fraud_detection',
    'anes_voting_2026',
    'lending_club',
    'musk',
    'amex_non_iid',
    'pancreatic_cancer_mouse_detection',
    'sepsis_prediction'
]


ABLATION_DATASETS_LEGACY = [
    "credit_approval",
    "heart_disease_cleveland",
    "hepatitis_survival_prediction",
    "musk",
    "blood_transfusion",
    "bank_marketing",
    "indian_liver_patient_dataset",
    "ljubljana_breast_cancer",
    "heart_failure_followup_survival",
    "early_stage_diabetes_risk_prediction",
]


def reproduce_dataset_partition():
    """Reproduce the deterministic BeyondArena dataset partition."""
    legacy = set(ABLATION_DATASETS_LEGACY)

    candidates = sorted(set(ALL_DATASETS) - legacy)

    rng = default_rng(DATASET_SPLIT_RANDOM_STATE)

    additional_ablation = sorted(
        rng.choice(
            candidates,
            size=15,
            replace=False,
        ).tolist()
    )

    ablation = sorted(
        [
            *ABLATION_DATASETS_LEGACY,
            *additional_ablation,
        ]
    )

    comparison = sorted(set(ALL_DATASETS) - set(ablation))

    return ablation, comparison


ABLATION_DATASETS = [
    'anes_voting_2026',
    'aps_failure',
    'bank_marketing',
    'blood_transfusion',
    'coil_2000',
    'credit_approval',
    'early_stage_diabetes_risk_prediction',
    'give_me_some_credit',
    'heart_disease_cleveland',
    'heart_failure_followup_survival',
    'hepatitis_survival_prediction',
    'homeq_default_prediction',
    'homesite_quote_conversion',
    'indian_liver_patient_dataset',
    'iranian_churn',
    'jm1',
    'ljubljana_breast_cancer',
    'lung_cancer_epithelial_genexp',
    'marketing_campaign',
    'musk',
    'pancreatic_cancer_mouse_detection',
    'pva_revenue_prediction_kddcup98',
    'sepsis_prediction',
    'south_africa_coronary_heart_disease',
    'wids_diabetes_mellitus'
]


COMPARISON_DATASETS = [
    'acquire_valued_shoppers_challenge',
    'amazon_employee_access',
    'amex_non_iid',
    'bad_customer_detection',
    'bank_customer_churn',
    'bioresponse',
    'churn',
    'clock_protein_toxicity',
    'credit_card_clients_default',
    'credit_g',
    'customer_satisfaction_in_airline',
    'diabetes_130_us',
    'drug_induced_autoimmunity_prediction',
    'ecommerce_shipping',
    'fitness_club',
    'gallstone_disease',
    'hazelnut_spread_contaminant_detection',
    'heart_disease_hungary',
    'heart_disease_va_long_beach',
    'heloc',
    'hiva_agnostic',
    'home_credit_default_risk',
    #'home_credit_default_stability', # This dataset was rejected because the FIGS algorithm crash the script
    'hotel_booking_demand',
    'hr_analytics',
    'ieee_fraud_detection',
    'in_vehicle_coupon_recommendation',
    'kdd_cup_09_appetency',
    'kick',
    'labour_inspection_compliance',
    'lending_club',
    'polish_companies_bankruptcy',
    'porto_seguro',
    'prostate_cancer_detection',
    'qsar_biodeg',
    'regensburg_pediatric_appendicitis',
    'santander_customer_satisfaction',
    'seismic_bumps',
    'taiwanese_bankruptcy_prediction',
    'thyroid_discordant',
    'tour_travels_churn'
]