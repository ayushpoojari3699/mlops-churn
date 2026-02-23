import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

MODEL_NAME = "churn-model"

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod"
]

X = df[FEATURES]
y = df["Churn"].map({"Yes": 1, "No": 0})

cat_cols = ["Contract", "InternetService", "PaymentMethod"]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("churn-mlops")

with mlflow.start_run(run_name="train"):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, name="model")

    joblib.dump(pipeline, MODEL_PATH)

    print("🎯 New model accuracy:", round(acc, 4))

    client = MlflowClient()

    # Register model
    result = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name=MODEL_NAME
    )

    # Compare with production model
    prod_acc = 0.0
    try:
        prod_ver = client.get_model_version_by_alias(MODEL_NAME, "production")
        prod_run = mlflow.get_run(prod_ver.run_id)
        prod_acc = prod_run.data.metrics.get("accuracy", 0.0)
    except Exception:
        print("ℹ️ No production model yet.")

    print("📊 Production accuracy:", prod_acc)
    print("📊 New model accuracy:", acc)

    if acc > prod_acc:
        client.set_registered_model_alias(MODEL_NAME, "production", result.version)
        print("🚀 Promoted new model to production alias!")
    else:
        print("❌ New model is not better. Production unchanged.")