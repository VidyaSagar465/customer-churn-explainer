import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# Configure Page
st.set_page_config(page_title="Customer Churn & Risk Explainer", layout="wide")

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    explainer = joblib.load("explainer.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, explainer, feature_cols

model, explainer, feature_cols = load_artifacts()

st.title("Customer Churn Risk & Explainability Engine")
st.markdown("Adjust customer attributes on the left to evaluate real-time churn probability and inspect SHAP risk factors.")

# Sidebar Controls for User Input
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = tenure * monthly_charges
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
paperless = st.sidebar.radio("Paperless Billing", ["Yes", "No"])

# Build single-row DataFrame matching trained feature set
raw_input = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    f"Contract_{contract}": True,
    f"InternetService_{internet_service}": True,
    "PaperlessBilling_Yes": True if paperless == "Yes" else False,
}

# Reindex to match training columns, filling missing dummy columns with False
input_df = pd.DataFrame([raw_input]).reindex(columns=feature_cols, fill_value=False)
input_df_float = input_df.astype(float)

# Compute Prediction & Probability
prob = model.predict_proba(input_df_float)[0][1]

# Render Prediction Header
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Churn Probability", value=f"{prob * 100:.1f}%")

with col2:
    if prob > 0.5:
        st.error("Status: High Risk Customer — Action Required")
    else:
        st.success("Status: Low Risk Customer")

# Render SHAP Waterfall Plot
st.subheader("Actionable Risk Breakdown (SHAP Waterfall Plot)")
shap_values = explainer(input_df_float)

fig, ax = plt.subplots(figsize=(8, 4))
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)