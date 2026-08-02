"""
Creates a small SYNTHETIC dataset shaped exactly like the real OULAD CSVs,
so you can smoke-test the whole pipeline in seconds without downloading
anything. This is NOT real student data -- replace it with
scripts/download_data.py before drawing any real conclusions.

Run directly:
    python scripts/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RNG = np.random.default_rng(42)

N_STUDENTS = 900
N_ENROLLMENTS = 1000  # some students appear more than once (retakes)
MODULES = ["AAA", "BBB", "CCC"]
PRESENTATIONS = ["2013J", "2014J"]
COURSE_LENGTH = 240  # days

REGIONS = ["East Anglian Region", "Scotland", "London Region", "North Western Region", "South East Region"]
EDUCATION = ["Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"]
IMD_BANDS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", None]
AGE_BANDS = ["0-35", "35-55", "55<="]


def make_assessment_calendar() -> pd.DataFrame:
    """4 assignments (early/mid) + 1 exam (late) per module/presentation, shared by every student in it."""
    rows, aid = [], 1
    for module in MODULES:
        for pres in PRESENTATIONS:
            for due_day in [20, 45, 90, 130]:
                rows.append({"code_module": module, "code_presentation": pres,
                             "id_assessment": aid, "assessment_type": "TMA", "date": due_day, "weight": 20})
                aid += 1
            rows.append({"code_module": module, "code_presentation": pres,
                         "id_assessment": aid, "assessment_type": "Exam", "date": 230, "weight": 20})
            aid += 1
    return pd.DataFrame(rows)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    student_pool = RNG.choice(np.arange(100000, 100000 + N_STUDENTS), size=N_STUDENTS, replace=False)
    enroll_student = RNG.choice(student_pool, size=N_ENROLLMENTS, replace=True)
    enroll_module = RNG.choice(MODULES, size=N_ENROLLMENTS)
    enroll_pres = RNG.choice(PRESENTATIONS, size=N_ENROLLMENTS)

    ability = RNG.normal(0, 1, size=N_ENROLLMENTS)  # latent engagement/preparedness, drives everything below

    gender = RNG.choice(["M", "F"], size=N_ENROLLMENTS)
    region = RNG.choice(REGIONS, size=N_ENROLLMENTS)
    highest_education = RNG.choice(EDUCATION, size=N_ENROLLMENTS)
    imd_band = RNG.choice(IMD_BANDS, size=N_ENROLLMENTS)
    age_band = RNG.choice(AGE_BANDS, size=N_ENROLLMENTS, p=[0.55, 0.35, 0.10])
    disability = RNG.choice(["N", "Y"], size=N_ENROLLMENTS, p=[0.9, 0.1])
    num_of_prev_attempts = RNG.choice([0, 0, 0, 1, 1, 2], size=N_ENROLLMENTS)
    studied_credits = RNG.choice([30, 60, 90, 120], size=N_ENROLLMENTS)
    date_registration = np.clip(RNG.normal(-30, 20, size=N_ENROLLMENTS), -120, 10).round()

    risk_logit = (-1.4 * ability - 0.3 * (disability == "Y") - 0.25 * num_of_prev_attempts
                  + RNG.normal(0, 0.6, N_ENROLLMENTS))
    at_risk = RNG.random(N_ENROLLMENTS) < (1 / (1 + np.exp(-risk_logit)))

    final_result = np.empty(N_ENROLLMENTS, dtype=object)
    for i in range(N_ENROLLMENTS):
        final_result[i] = (RNG.choice(["Fail", "Withdrawn"], p=[0.55, 0.45]) if at_risk[i]
                            else RNG.choice(["Pass", "Distinction"], p=[0.75, 0.25]))

    date_unregistration = np.full(N_ENROLLMENTS, np.nan)
    withdrawn_mask = final_result == "Withdrawn"
    date_unregistration[withdrawn_mask] = RNG.integers(5, 230, size=withdrawn_mask.sum())

    student_info = pd.DataFrame({
        "code_module": enroll_module, "code_presentation": enroll_pres, "id_student": enroll_student,
        "gender": gender, "region": region, "highest_education": highest_education, "imd_band": imd_band,
        "age_band": age_band, "num_of_prev_attempts": num_of_prev_attempts, "studied_credits": studied_credits,
        "disability": disability, "final_result": final_result,
    }).drop_duplicates(subset=["code_module", "code_presentation", "id_student"])

    student_reg = pd.DataFrame({
        "code_module": enroll_module, "code_presentation": enroll_pres, "id_student": enroll_student,
        "date_registration": date_registration, "date_unregistration": date_unregistration,
    }).drop_duplicates(subset=["code_module", "code_presentation", "id_student"])

    keep_idx = student_info.index
    ability = ability[keep_idx]
    stu = student_info["id_student"].to_numpy()
    mod = student_info["code_module"].to_numpy()
    pres = student_info["code_presentation"].to_numpy()
    n = len(student_info)

    vle_rows, site_id = [], 5000
    for i in range(n):
        n_events = int(RNG.poisson(lam=max(0.5, 8 + 6 * ability[i])))
        if n_events == 0:
            continue
        days = RNG.integers(0, COURSE_LENGTH, size=n_events)
        clicks = np.clip(RNG.poisson(lam=max(0.5, 4 + 3 * ability[i]), size=n_events), 1, None)
        for d, c in zip(days, clicks):
            vle_rows.append({"code_module": mod[i], "code_presentation": pres[i], "id_student": stu[i],
                             "id_site": site_id, "date": int(d), "sum_click": int(c)})
        site_id += 1
    student_vle = pd.DataFrame(vle_rows)

    assessments = make_assessment_calendar()
    tma_only = assessments[assessments["assessment_type"] == "TMA"]
    sa_rows = []
    for i in range(n):
        mod_tmas = tma_only[(tma_only["code_module"] == mod[i]) & (tma_only["code_presentation"] == pres[i])]
        for _, a in mod_tmas.iterrows():
            if RNG.random() > np.clip(0.55 + 0.3 * ability[i], 0.05, 0.98):
                continue
            score = np.clip(RNG.normal(60 + 15 * ability[i], 12), 0, 100)
            lateness = max(0, int(RNG.normal(1, 3)))
            sa_rows.append({"id_assessment": a["id_assessment"], "id_student": stu[i],
                            "date_submitted": int(a["date"] + lateness), "is_banked": 0, "score": round(float(score), 1)})
    student_assessment = pd.DataFrame(sa_rows)

    student_info.to_csv(RAW_DIR / "studentInfo.csv", index=False)
    student_reg.to_csv(RAW_DIR / "studentRegistration.csv", index=False)
    assessments.to_csv(RAW_DIR / "assessments.csv", index=False)
    student_assessment.to_csv(RAW_DIR / "studentAssessment.csv", index=False)
    student_vle.to_csv(RAW_DIR / "studentVle.csv", index=False)

    at_risk_rate = student_info["final_result"].isin(["Fail", "Withdrawn"]).mean()
    print(f"Synthetic data written to {RAW_DIR}")
    print(f"  studentInfo.csv:        {len(student_info):>5} rows  (at-risk rate {at_risk_rate:.1%})")
    print(f"  studentRegistration.csv:{len(student_reg):>5} rows")
    print(f"  assessments.csv:        {len(assessments):>5} rows")
    print(f"  studentAssessment.csv:  {len(student_assessment):>5} rows")
    print(f"  studentVle.csv:         {len(student_vle):>5} rows")


if __name__ == "__main__":
    main()