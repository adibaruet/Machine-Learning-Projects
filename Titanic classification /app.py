from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat
from joblib import load
import numpy as np

app = FastAPI(title="Titanic Survival API", version="1.0")

# Load trained model
model = load("titanic_model.joblib")

# Input schema for passenger details
class Passenger(BaseModel):
    Pclass: conint(ge=1, le=3) = Field(..., description="Passenger class: 1, 2, or 3")
    Sex: str = Field(..., description="male or female")
    Age: confloat(ge=0, le=100) = Field(..., description="Age in years")
    SibSp: conint(ge=0) = Field(..., description="# of siblings/spouses aboard")
    Parch: conint(ge=0) = Field(..., description="# of parents/children aboard")
    Fare: confloat(ge=0) = Field(..., description="Ticket fare")
    Embarked: str = Field(..., description="C, Q, or S")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(p: Passenger):
    row = {
        "Pclass": p.Pclass,
        "Sex": p.Sex.lower(),
        "Age": p.Age,
        "SibSp": p.SibSp,
        "Parch": p.Parch,
        "Fare": p.Fare,
        "Embarked": p.Embarked.upper()
    }
    proba = model.predict_proba([row])[0][1]
    pred = int(proba >= 0.5)

    return {
        "survived": pred,
        "probability_survive": float(np.round(proba, 4)),
        "threshold": 0.5
    }
