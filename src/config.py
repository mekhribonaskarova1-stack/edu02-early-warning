"""
Central place for every setting used across the project.
Change a value here once, and every script picks it up.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

FEATURES_PATH = PROCESSED_DIR / "features.csv"
TEST_IDS_PATH = PROCESSED_DIR / "test_ids.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

CUTOFF_DAY = 60
EXCLUDED_ASSESSMENT_TYPES = {"Exam"}
AT_RISK_RESULTS = {"Fail", "Withdrawn"}
NOT_AT_RISK_RESULTS = {"Pass", "Distinction"}

RANDOM_STATE = 42
TEST_SIZE = 0.2
GROUP_COLUMN = "id_student"

CATEGORICAL_FEATURES = [
    "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "disability",
]

NUMERIC_FEATURES = [
    "num_of_prev_attempts", "studied_credits", "date_registration",
    "withdrawn_before_cutoff", "sum_click_total", "n_active_days",
    "n_early_assessments", "mean_early_score", "min_early_score",
]

TARGET_COLUMN = "at_risk"
ID_COLUMNS = ["id_student", "code_module", "code_presentation"]

RISK_BANDS = [
    (0.0, 0.3, "Low"),
    (0.3, 0.6, "Medium"),
    (0.6, 1.01, "High"),
]