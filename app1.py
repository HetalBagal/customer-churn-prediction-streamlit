import streamlit as st
import pickle
import pandas as pd


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------- LOAD MODEL ----------
model = pickle.load(open("model/model.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.card {
    background-color: #1E1E2F;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
.kpi {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Analyze customer behavior and predict churn risk")

st.divider()

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Prediction", "ℹ️ About"])

# ---------- SIDEBAR ----------
st.sidebar.header("🧾 Customer Input")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", value=70.0)
total_charges = st.sidebar.number_input("Total Charges", value=1000.0)

gender = st.sidebar.selectbox("Gender", encoders["gender"].classes_)
partner = st.sidebar.selectbox("Partner", encoders["Partner"].classes_)
dependents = st.sidebar.selectbox("Dependents", encoders["Dependents"].classes_)

phone_service = st.sidebar.selectbox("Phone Service", encoders["PhoneService"].classes_)
internet_service = st.sidebar.selectbox("Internet Service", encoders["InternetService"].classes_)
contract = st.sidebar.selectbox("Contract Type", encoders["Contract"].classes_)
payment_method = st.sidebar.selectbox("Payment Method", encoders["PaymentMethod"].classes_)
paperless_billing = st.sidebar.selectbox("Paperless Billing", encoders["PaperlessBilling"].classes_)

# ---------- DASHBOARD ----------
with tab1:
    st.subheader("📊 Customer Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
        <h3>📋 Customer Summary</h3>
        <p><b>Tenure:</b> {tenure} months</p>
        <p><b>Monthly Charges:</b> ₹{monthly_charges}</p>
        <p><b>Contract:</b> {contract}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        chart_data = pd.DataFrame({
            "Feature": ["Tenure", "Monthly Charges"],
            "Value": [tenure, monthly_charges]
        })
        st.bar_chart(chart_data.set_index("Feature"))

# ---------- PREDICTION ----------
with tab2:
    st.subheader("🔍 Predict Customer Churn")

    if st.button("🚀 Run Prediction", use_container_width=True):

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

        # Encode
        for column in input_df.columns:
            if column in encoders:
                input_df[column] = encoders[column].transform(input_df[column])

        input_df = input_df[columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("📢 Prediction Result")

        # KPI CARDS
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"""
        <div class="kpi" style="background-color:#FF4B4B;">
        <h4>Risk</h4>
        <h2>{probability*100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="kpi" style="background-color:#00C897;">
        <h4>Tenure</h4>
        <h2>{tenure}</h2>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="kpi" style="background-color:#4D96FF;">
        <h4>Charges</h4>
        <h2>₹{monthly_charges}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(probability))

        # RESULT
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

        # INSIGHTS
        st.markdown("### 📌 Insights")

        if probability > 0.7:
            st.error("🚨 High Risk — Immediate action needed")
        elif probability > 0.4:
            st.warning("⚠️ Moderate Risk — Monitor customer")
        else:
            st.success("✅ Low Risk — Stable customer")

# ---------- ABOUT ----------
with tab3:
    st.subheader("ℹ️ About Project")

    st.markdown("""
    ### 📊 Customer Churn Prediction System

    This app predicts whether a customer will churn using ML.

    ### 🚀 Features:
    - Interactive Dashboard
    - Real-time Prediction
    - Risk Analysis

    ### 🛠 Tech Stack:
    - Python
    - Streamlit
    - Machine Learning

    ---
    💻 Developed by Hetal
    """)