import pandas as pd
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "data", "predictions_log.csv")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "churn_clean.csv")

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
DRIFT_THRESHOLD = 0.3

def detect_drift(train_df, live_df):
    drifted = []
    for col in NUM_FEATURES:
        train_mean = train_df[col].mean()
        live_mean = live_df[col].mean()
        if train_mean == 0:
            continue
        relative_change = abs(live_mean - train_mean) / abs(train_mean)
        if relative_change > DRIFT_THRESHOLD:
            drifted.append((col, relative_change))
    return drifted

def trigger_retraining():
    print("🚨 Drift detected! Triggering retraining...")
    subprocess.run(["python", os.path.join(BASE_DIR, "src", "train.py")])

def main():
    if not os.path.exists(LOG_PATH):
        print("No predictions yet.")
        return

    live_df = pd.read_csv(LOG_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    drifted = detect_drift(train_df, live_df)

    if drifted:
        print("⚠️ Drift detected in:")
        for col, change in drifted:
            print(f" - {col}: {round(change*100, 2)}%")
        trigger_retraining()
    else:
        print("✅ No significant drift detected.")

if __name__ == "__main__":
    main()