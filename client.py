import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "tenure": 5,
    "MonthlyCharges": 70.5,
    "TotalCharges": 350.2,
    "Contract": "One year",
    "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)"
}

res = requests.post(url, json=payload)

print("Status code:", res.status_code)
print("Raw response:")
print(res.text)   # 👈 print raw response first