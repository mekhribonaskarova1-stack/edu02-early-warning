"""
Pydantic models describing exactly what /predict accepts and returns.
"""
from typing import Optional
from pydantic import BaseModel, Field


class StudentSnapshot(BaseModel):
    code_module: str = Field(..., examples=["BBB"])
    code_presentation: str = Field(..., examples=["2013J"])
    gender: str = Field(..., examples=["F"])
    region: str = Field(..., examples=["East Anglian Region"])
    highest_education: str = Field(..., examples=["A Level or Equivalent"])
    imd_band: Optional[str] = Field(None, examples=["30-40%"])
    age_band: str = Field(..., examples=["0-35"])
    disability: str = Field(..., examples=["N"])
    num_of_prev_attempts: int = Field(..., ge=0, examples=[0])
    studied_credits: int = Field(..., ge=0, examples=[60])
    date_registration: Optional[float] = Field(None, examples=[-45])
    withdrawn_before_cutoff: int = Field(0, ge=0, le=1, examples=[0])
    sum_click_total: float = Field(..., ge=0, examples=[820])
    n_active_days: float = Field(..., ge=0, examples=[27])
    n_early_assessments: float = Field(..., ge=0, examples=[2])
    mean_early_score: Optional[float] = Field(None, examples=[68.5])
    min_early_score: Optional[float] = Field(None, examples=[55])


class RiskPrediction(BaseModel):
    at_risk_probability: float
    prediction: int
    risk_band: str