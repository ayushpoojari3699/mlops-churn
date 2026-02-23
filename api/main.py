from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime
import time
from threading import Thread
import joblib

USE_MLFLOW = os.getenv("USE_MLFLOW", "false").lower() == "true"

if USE_MLFLOW:
    import mlflow.sklearn

app = FastAPI(title="Churn Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "data", "predictions_log.csv")

MODEL_NAME = "churn-model"
MODEL_ALIAS = "production"
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

pipeline = None


def load_model():
    global pipeline
    if USE_MLFLOW:
        pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        print("✅ Loaded model from MLflow Registry")
    else:
        pipeline = joblib.load(LOCAL_MODEL_PATH)
        print("✅ Loaded local model:", LOCAL_MODEL_PATH)


# Initial load
load_model()


# Auto reload every 60 seconds
def reload_model_loop():
    while True:
        time.sleep(60)
        try:
            load_model()
        except Exception as e:
            print("⚠️ Reload failed:", e)


Thread(target=reload_model_loop, daemon=True).start()


class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str


@app.get("/")
def home():
    return {"status": "API is running", "use_mlflow": USE_MLFLOW}


@app.post("/predict")
def predict_churn(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])

    prob = float(pipeline.predict_proba(data)[0][1])

    THRESHOLD = 0.3
    pred = int(prob > THRESHOLD)

    log_row = data.copy()
    log_row["prediction"] = pred
    log_row["probability"] = prob
    log_row["timestamp"] = datetime.utcnow().isoformat()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if not os.path.exists(LOG_PATH):
        log_row.to_csv(LOG_PATH, index=False)
    else:
        log_row.to_csv(LOG_PATH, mode="a", header=False, index=False)

    return {
        "churn": bool(pred),
        "churn_probability": round(prob, 4),
        "threshold_used": THRESHOLD
    }