import streamlit as st
import pickle
import pandas as pd

# Load model artifacts
model = pickle.load(open("model/model.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("### Predict whether a customer is likely to churn")

st.divider()

# ---------- Layout ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")

    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", value=70.0)
    total_charges = st.number_input("Total Charges", value=1000.0)

    gender = st.selectbox("Gender", encoders["gender"].classes_)
    partner = st.selectbox("Partner", encoders["Partner"].classes_)
    dependents = st.selectbox("Dependents", encoders["Dependents"].classes_)

with col2:
    st.subheader("Service Information")

    phone_service = st.selectbox("Phone Service", encoders["PhoneService"].classes_)
    internet_service = st.selectbox("Internet Service", encoders["InternetService"].classes_)
    contract = st.selectbox("Contract Type", encoders["Contract"].classes_)
    payment_method = st.selectbox("Payment Method", encoders["PaymentMethod"].classes_)
    paperless_billing = st.selectbox("Paperless Billing", encoders["PaperlessBilling"].classes_)

st.divider()

# ---------- Prediction ----------
if st.button("🔍 Predict Churn"):

    input_dict = {
        "gender": gender,
        "SeniorCitizen": 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical features safely
    for column in input_df.columns:
        if column in encoders:
            input_df[column] = encoders[column].transform(input_df[column])

    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to Churn")
    else:
        st.success(f"✅ Customer is likely to Stay")

    st.metric("Churn Probability", f"{probability*100:.2f}%")

    st.progress(float(probability))

    if probability > 0.7:
        st.warning("High Risk Customer — Immediate retention action recommended.")
    elif probability > 0.4:
        st.info("Moderate Risk — Monitor customer closely.")
    else:
        st.success("Low Risk — Customer is stable.")
