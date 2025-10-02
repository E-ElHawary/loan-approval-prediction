import joblib
import pandas as pd
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, Field
from typing import Literal
from constants import MODELS_PATH, DATA_PATH

# ---------- Load Data and Model Once ----------
df = pd.read_csv(DATA_PATH / "loan_approval_dataset.csv")
model = joblib.load(MODELS_PATH / "rf_pipeline.joblib")

# ---------- App Setup ----------
app = FastAPI(title="Loan Approval Prediction API", version="1.0.0")
router = APIRouter(prefix="/api/loan/v1", tags=["predictions"])

# ---------- Request Schema ----------
class LoanRecord(BaseModel):
    no_of_dependents: int = Field(..., ge=0, le=5)
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"] = Field(..., alias="selfEmployed")
    income_annum: float = Field(..., alias="incomeAnnum", ge=200_000, le=9_900_000)
    loan_amount: float = Field(..., alias="loanAmount", ge=300_000, le=39_500_000)
    loan_term: float = Field(..., alias="loanTerm", ge=2, le=20)
    cibil_score: float = Field(..., alias="cibilScore", ge=300, le=900)
    residential_assets_value: float = Field(..., alias="residentialAssetsValue", ge=-100_000, le=29_100_000)
    commercial_assets_value: float = Field(..., alias="commercialAssetsValue", ge=0, le=19_400_000)
    luxury_assets_value: float = Field(..., alias="luxuryAssetsValue", ge=300_000, le=39_200_000)
    bank_asset_value: float = Field(..., alias="bankAssetValue", ge=0, le=14_700_000)

# ---------- Response Schema ----------
class PredictionResponse(BaseModel):
    approved: str
    probability: float

# ---------- Routes ----------
@app.get("/")
def read_data():
    """Return a preview of the dataset."""
    return df.head(5).to_dict(orient="records")

@router.post("/predict", response_model=PredictionResponse)
def predict_loan_approval(payload: LoanRecord):
    """Predict loan approval from applicant data."""
    data_to_predict = pd.DataFrame(payload.model_dump(), index=[0])

    # Prediction
    prediction = model.predict(data_to_predict)[0]
    proba = model.predict_proba(data_to_predict)[0]

    # Match probability to predicted class
    probability = float(proba[model.classes_.tolist().index(prediction)])

    return PredictionResponse(approved=str(prediction), probability=probability)

# ---------- Include Router ----------
app.include_router(router)
