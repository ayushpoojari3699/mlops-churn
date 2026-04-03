# MLOps Churn Prediction System

##  Overview

This project implements an **end-to-end MLOps pipeline** for customer churn prediction. It includes:

*  Machine Learning model for churn prediction
*  FastAPI backend for inference
*  Streamlit UI for user interaction
*  Dockerized deployment
*  Production deployment via Railway

---

##  Project Architecture

```id="arch01"
mlops-churn/
│
├── api/                  # FastAPI backend
├── src/                  # Training & preprocessing
├── models/               # Saved ML models (ignored in git)
├── data/                 # Dataset (ignored in git)
├── mlruns/               # MLflow tracking (ignored)
│
├── ui/                   # Streamlit frontend
│   └── app.py
│
├── client.py             # API testing script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

##  Features

* Customer churn prediction using ML
*  REST API using FastAPI
*  MLflow experiment tracking
* Interactive UI using Streamlit
*  Docker support for reproducibility
*  Deployed on Railway

---

##  Setup Instructions

### 1️ Clone Repository

```id="clone01"
git clone https://github.com/ayushpoojari3699/mlops-churn.git
cd mlops-churn
```

---

## Running the API

### Local

```id="api01"
uvicorn api.main:app --reload
```

API available at:

```id="api02"
http://127.0.0.1:8000
```

Swagger Docs:

```id="api03"
http://127.0.0.1:8000/docs
```

---

### Production (Railway)

```id="api04"
https://mlops-churn-production.up.railway.app
```

Swagger:

```id="api05"
https://mlops-churn-production.up.railway.app/docs
```

---

## Running the UI (Streamlit)

### Install dependencies

```id="ui01"
pip install streamlit requests
```

### Run UI

```id="ui02"
cd ui
streamlit run app.py
```

App opens at:

```id="ui03"
http://localhost:8501
```

---

##  API Configuration

Edit `ui/app.py`:

```python id="cfg01"
# Local
# API_URL = "http://127.0.0.1:8000/predict"

# Production
API_URL = "https://mlops-churn-production.up.railway.app/predict"
```

---

##  API Usage Example

```python id="api_test01"
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
print(res.text)
```

---

##  Example Input

| Feature         | Value            |
| --------------- | ---------------- |
| Tenure          | 41               |
| MonthlyCharges  | 70               |
| TotalCharges    | 350              |
| Contract        | Month-to-month   |
| InternetService | Fiber optic      |
| PaymentMethod   | Electronic check |

---

##  Docker Support

### Build & Run

```id="docker01"
docker-compose up --build
```

---

##  Model & Data

Large files are excluded from the repository.

Add manually:

```id="model01"
models/
data/
mlruns/
```

---

##  Notes

* Ensure model file is present before running API
* Update API_URL in UI before deployment
* Do not commit sensitive credentials

---

##  Future Improvements

* CI/CD pipeline (GitHub Actions)
* Kubernetes deployment
* Model monitoring (drift detection)
* Authentication for API
* Real-time streaming predictions

---

## Author

**Ayush Poojari**
GitHub: https://github.com/ayushpoojari3699


