# EDU-02 — Student Performance Early-Warning Model

A day-60 early-warning system: given only what's known about a student
60 days into a course, estimate the probability they'll fail or withdraw,
so academic support staff can step in while there's still time.

---

## 1. Dataset

**Source:** Open University Learning Analytics Dataset (OULAD) — https://analyse.kmi.open.ac.uk/open_dataset
**License:** CC-BY 4.0 (Kuzilek, Hlosta & Zdrahal, 2017)

OULAD contains ~32,000 enrollments across 7 modules, with demographics,
registration dates, VLE (virtual learning environment) clickstream logs,
and assessment submissions/scores. It's anonymized and was released
specifically for learning-analytics research, which is why it's a
reasonable public stand-in for a real institution's data.

## 2. Data & Problem Discovery

| Question | Answer |
|---|---|
| **What does one record represent?** | One student's attempt at one module presentation (a student who retakes a module has two records). |
| **Target** | Binary: `at_risk` = 1 if `final_result` is `Fail` or `Withdrawn`, 0 if `Pass` or `Distinction`. |
| **Prediction point** | Day 60 of the course (`config.CUTOFF_DAY`). Chosen because most modules run ~240 days, so day 60 is early enough to still act on, but late enough that some real engagement signal exists. |
| **Info available at day 60** | Demographics, registration date, whether the student already withdrew, VLE clicks/active days up to day 60, and scores on early assignments due and submitted by day 60. |
| **Main data quality issues** | `imd_band` (deprivation index) has missing values; the dataset covers one UK institution only, so it may not generalize elsewhere; VLE click counts measure activity, not comprehension. |
| **Leakage risks & how they're avoided** | Final exam scores, late or not-yet-submitted assignments, and any VLE activity after day 60 are excluded by date filtering in `src/data_prep.py`. Assessment type `Exam` is excluded outright, even if a date glitch put it before day 60. |
| **Privacy / fairness concerns** | Demographic fields (`disability`, `region`, `imd_band`) are known risk correlates in education research — using them can improve accuracy but also risks encoding existing inequities into who gets flagged. See §10 Limitations. |

## 3. Technical Proposal

| Question | Answer |
|---|---|
| **ML formulation** | Binary classification (`at_risk` probability), not regression — support staff need a "who to check on," not a precise grade forecast. |
| **Baseline** | Logistic Regression, `class_weight="balanced"`. |
| **Main model** | `HistGradientBoostingClassifier` — handles the mix of categorical/numeric features and non-linear interactions well, at this dataset size. |
| **Validation strategy** | `GroupShuffleSplit` grouped by `id_student`, 80/20. Grouping matters because a student who retakes a module could otherwise leak into both train and test. |
| **Primary metrics** | ROC-AUC and PR-AUC (the classes are imbalanced — most students pass). We also report recall/precision at a lowered decision threshold (0.3), because missing an at-risk student is more costly than one unnecessary check-in, so we deliberately favor recall over precision. |
| **Inference input** | One student's day-60 snapshot (see `src/inference_schema.py`). |
| **Inference output** | `at_risk_probability`, a binary `prediction`, and a `risk_band` (Low/Medium/High) that's easier for a non-technical advisor to act on. |
| **Main risks / assumptions** | Assumes day-60 engagement patterns are stable predictors across modules; a model trained on one module mix may drift if the institution's course catalog changes. |

## 4. Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

### Get the dataset

    python scripts/download_data.py      # Option A: real OULAD data (needs internet)
    python scripts/make_sample_data.py   # Option B: fast synthetic stand-in

## 5. Run the pipeline

    python -m src.data_prep
    python -m src.train
    python -m src.evaluate

## 6. Run the API

    uvicorn api.main:app --reload

Then in another terminal:

    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{...}'

Or open http://127.0.0.1:8000/docs for an interactive form.

## 7. Run the tests

    pytest -v

## 8. Results (on the bundled synthetic sample)

| Model | ROC-AUC | PR-AUC | Recall @ 0.3 | Precision @ 0.3 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.76 | 0.70 | 85% | 54% |
| HistGradientBoostingClassifier (main) | 0.74 | 0.71 | 76% | 56% |

Synthetic data is for testing the code, not for drawing conclusions about
real students — re-run on the real OULAD data before trusting any number.

## 9. Limitations, risks & next steps

- Fairness: compare false-negative rates across region/disability/imd_band
  groups before deploying, not just overall accuracy.
- Single-institution data: OULAD reflects one UK setup; may not transfer.
- Engagement ≠ understanding: VLE clicks measure activity, not comprehension.
- Next steps: explainable risk factors (SHAP), temporal evaluation at other
  cutoffs, a simple advisor-facing dashboard.