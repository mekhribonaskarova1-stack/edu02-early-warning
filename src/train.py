"""
Trains the baseline model and the main model, evaluates both on a
held-out group split, and saves the winner.

Run: python -m src.train
"""
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src import config


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, config.NUMERIC_FEATURES),
        ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
    ])


def recall_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "flagged_rate": (y_pred == 1).mean()}


def evaluate(model, X_test, y_test, label: str) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "report_at_0.5": classification_report(y_test, y_pred, output_dict=True),
        "recall_at_0.3": recall_at_threshold(y_test, y_proba, 0.3),
    }

    print(f"\n--- {label} ---")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}   PR-AUC: {metrics['pr_auc']:.3f}")
    r = metrics["recall_at_0.3"]
    print(
        f"At threshold 0.3 -> recall {r['recall']:.1%}, precision {r['precision']:.1%}, "
        f"flags {r['flagged_rate']:.1%} of students"
    )
    return metrics


def main():
    df = pd.read_csv(config.FEATURES_PATH)

    X = df[config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES]
    y = df[config.TARGET_COLUMN]
    groups = df[config.GROUP_COLUMN]

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    print(f"Students appearing in both train and test: {len(overlap)} (must be 0)")

    baseline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test, "Baseline: Logistic Regression")

    main_model = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(random_state=config.RANDOM_STATE, class_weight="balanced")),
    ])
    main_model.fit(X_train, y_train)
    main_metrics = evaluate(main_model, X_test, y_test, "Main model: HistGradientBoostingClassifier")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(main_model, config.MODEL_PATH)
    joblib.dump(baseline, config.BASELINE_MODEL_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump({"baseline": baseline_metrics, "main_model": main_metrics}, f, indent=2, default=float)

    df.iloc[test_idx][config.ID_COLUMNS].to_csv(config.TEST_IDS_PATH, index=False)

    print(f"\nSaved main model -> {config.MODEL_PATH}")
    print(f"Saved baseline model -> {config.BASELINE_MODEL_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}")
    print(f"Saved held-out test IDs -> {config.TEST_IDS_PATH}")


if __name__ == "__main__":
    main()