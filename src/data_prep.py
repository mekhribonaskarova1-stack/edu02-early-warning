"""
Builds the modeling table from the raw OULAD CSV files.
Every feature uses ONLY information that would have existed by
config.CUTOFF_DAY of the course.

Run: python -m src.data_prep
"""
import sys
import pandas as pd

from src import config


def _require_raw_files() -> None:
    needed = [
        "studentInfo.csv", "studentRegistration.csv", "assessments.csv",
        "studentAssessment.csv", "studentVle.csv",
    ]
    missing = [f for f in needed if not (config.RAW_DIR / f).exists()]
    if missing:
        print(
            "Missing raw data file(s): " + ", ".join(missing) +
            f"\nExpected them in: {config.RAW_DIR}\n"
            "Run 'python scripts/download_data.py' first, or see the "
            "README's 'Get the dataset' section.",
            file=sys.stderr,
        )
        sys.exit(1)


def load_raw():
    _require_raw_files()
    student_info = pd.read_csv(config.RAW_DIR / "studentInfo.csv")
    student_reg = pd.read_csv(config.RAW_DIR / "studentRegistration.csv")
    assessments = pd.read_csv(config.RAW_DIR / "assessments.csv")
    student_assessment = pd.read_csv(config.RAW_DIR / "studentAssessment.csv")
    student_vle = pd.read_csv(config.RAW_DIR / "studentVle.csv")
    return student_info, student_reg, assessments, student_assessment, student_vle


def build_base_table(student_info: pd.DataFrame, student_reg: pd.DataFrame) -> pd.DataFrame:
    base = student_info.merge(
        student_reg, on=["code_module", "code_presentation", "id_student"], how="left",
    )
    base = base[base["final_result"].notna()].copy()
    base[config.TARGET_COLUMN] = base["final_result"].apply(
        lambda r: 1 if r in config.AT_RISK_RESULTS else 0
    )
    base["withdrawn_before_cutoff"] = (
        base["date_unregistration"].notna()
        & (base["date_unregistration"] <= config.CUTOFF_DAY)
    ).astype(int)
    return base


def build_vle_features(student_vle: pd.DataFrame) -> pd.DataFrame:
    early = student_vle[student_vle["date"] <= config.CUTOFF_DAY]
    agg = (
        early.groupby(["code_module", "code_presentation", "id_student"])
        .agg(sum_click_total=("sum_click", "sum"), n_active_days=("date", "nunique"))
        .reset_index()
    )
    return agg


def build_assessment_features(assessments: pd.DataFrame, student_assessment: pd.DataFrame) -> pd.DataFrame:
    usable_assessments = assessments[
        (assessments["date"].notna())
        & (assessments["date"] <= config.CUTOFF_DAY)
        & (~assessments["assessment_type"].isin(config.EXCLUDED_ASSESSMENT_TYPES))
    ]
    merged = student_assessment.merge(usable_assessments, on="id_assessment", how="inner")
    merged = merged[merged["date_submitted"] <= config.CUTOFF_DAY]
    agg = (
        merged.groupby(["code_module", "code_presentation", "id_student"])
        .agg(
            n_early_assessments=("score", "count"),
            mean_early_score=("score", "mean"),
            min_early_score=("score", "min"),
        )
        .reset_index()
    )
    return agg


def build_features() -> pd.DataFrame:
    student_info, student_reg, assessments, student_assessment, student_vle = load_raw()

    base = build_base_table(student_info, student_reg)
    vle_feats = build_vle_features(student_vle)
    assess_feats = build_assessment_features(assessments, student_assessment)

    df = base.merge(vle_feats, on=["code_module", "code_presentation", "id_student"], how="left")
    df = df.merge(assess_feats, on=["code_module", "code_presentation", "id_student"], how="left")

    df["sum_click_total"] = df["sum_click_total"].fillna(0)
    df["n_active_days"] = df["n_active_days"].fillna(0)
    df["n_early_assessments"] = df["n_early_assessments"].fillna(0)
    # mean/min_early_score stay NaN on purpose -- the training imputer handles that

    keep_cols = config.ID_COLUMNS + [config.TARGET_COLUMN] + config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES
    keep_cols = list(dict.fromkeys(keep_cols))
    df = df[keep_cols]
    return df


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_features()
    df.to_csv(config.FEATURES_PATH, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns -> {config.FEATURES_PATH}")
    print(f"At-risk rate: {df[config.TARGET_COLUMN].mean():.1%}")


if __name__ == "__main__":
    main()