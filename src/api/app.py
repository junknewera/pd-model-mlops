import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

THRESHOLD = 0.5
state = {}


class Client(BaseModel):
    LIMIT_BAL: float = Field(gt=0, le=10_000_000)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class Prediction(BaseModel):
    default: int
    probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(os.getenv("MODEL_PATH", "models/model.joblib"))
    yield
    state.clear()


app = FastAPI(title="PD model API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(client: Client):
    X = pd.DataFrame([client.model_dump()])
    proba = float(state["model"].predict_proba(X)[0, 1])
    return Prediction(default=int(proba >= THRESHOLD), probability=proba)
