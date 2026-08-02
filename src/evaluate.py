"""
Reloads the saved model and the exact held-out test rows train.py set
aside, then reprints the evaluation -- proves the saved model.joblib
really is the one that produced the reported numbers.

Run: python -m src.evaluate
"""
import pandas as pd
import joblib

from src import config
from src.train import evaluate


def main():
    df = pd.read_csv(config.FEATURES_PATH)
    test_ids = pd.read_csv(config.TEST_IDS_PATH)

    test_df = df.merge(test_ids, on=config.ID_COLUMNS, how="inner")
    X_test = test_df[config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES]
    y_test = test_df[config.TARGET_COLUMN]

    model = joblib.load(config.MODEL_PATH)
    evaluate(model, X_test, y_test, "Reloaded main model on saved held-out set")


if __name__ == "__main__":
    main()