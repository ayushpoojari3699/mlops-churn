import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📉 Customer Churn Predictor")

with st.form("churn_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=350.0)

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    submitted = st.form_submit_button("Predict Churn 🚀")

if submitted:
    payload = {
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
        "Contract": Contract,
        "InternetService": InternetService,
        "PaymentMethod": PaymentMethod
    }

    with st.spinner("Calling ML API..."):
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if res.status_code == 200:
        data = res.json()
        st.success("Prediction received!")
        st.metric("Churn Probability", f"{data['churn_probability']*100:.2f}%")
        st.write("Churn:", "❌ Yes" if data["churn"] else "✅ No")
        st.caption(f"Threshold used: {data['threshold_used']}")
    else:
        st.error(f"API Error {res.status_code}")
        st.text(res.text)