import pandas as pd

df = pd.read_csv("data/churn.csv")

# Fix TotalCharges (some values are blank strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Save cleaned data
df.to_csv("data/churn_clean.csv", index=False)

print("Cleaned data saved as data/churn_clean.csv")