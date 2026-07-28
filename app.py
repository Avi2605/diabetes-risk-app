import time

import joblib
import pandas as pd
import streamlit as st

# ---- Load artifacts (must sit in the same folder as this file) ----
model = joblib.load("xgb_diabetes_model.pkl")
scaler = joblib.load("robust_scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

CLASS_LABELS = {0: "Negative", 1: "Prediabetes", 2: "Type 2 Diabetes"}
FEATURE_ORDER = ["Age", "Gender", "BMI", "Systolic_BP", "Diastolic_BP", "HbA1c_Level"]
SCALE_COLS = ["Age", "BMI", "Systolic_BP", "Diastolic_BP", "HbA1c_Level"]
BP_GLITCH_THRESHOLD = 20

DEFAULTS = {
    "age": 30,
    "gender": "Male",
    "bmi": 24.0,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "hba1c": 5.5,
}

# Initialize session state once, so widgets have somewhere to read/reset from
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.set_page_config(page_title="Diabetes Risk Checker", page_icon="🩺")
st.title("🩺 Diabetes Risk Checker")
st.caption("Powered by an XGBoost classifier trained on patient biomarkers.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, key="age")
        gender = st.radio(
            "Gender",
            options=list(gender_encoder.classes_),
            horizontal=True,
            key="gender",
        )
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, key="bmi")
    with col2:
        systolic_bp = st.number_input(
            "Systolic BP (mmHg)", min_value=60, max_value=250, key="systolic_bp"
        )
        diastolic_bp = st.number_input(
            "Diastolic BP (mmHg)", min_value=40, max_value=150, key="diastolic_bp"
        )
        hba1c = st.number_input(
            "HbA1c Level (%)", min_value=3.0, max_value=20.0, key="hba1c"
        )

    submitted = st.form_submit_button("Check Risk")

# # Reset button lives outside the form, so it can clear it independently
# if st.button("Reset"):
#     for key, value in DEFAULTS.items():
#         st.session_state[key] = value
#     st.rerun()

if submitted:
    if systolic_bp < BP_GLITCH_THRESHOLD or diastolic_bp < BP_GLITCH_THRESHOLD:
        st.error("Blood pressure values look invalid.")
    else:
        with st.spinner("Running prediction..."):
            gender_encoded = gender_encoder.transform([gender])[0]
            input_df = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Gender": gender_encoded,
                        "BMI": bmi,
                        "Systolic_BP": systolic_bp,
                        "Diastolic_BP": diastolic_bp,
                        "HbA1c_Level": hba1c,
                    }
                ]
            )[FEATURE_ORDER]

            input_scaled = input_df.copy()
            input_scaled[SCALE_COLS] = scaler.transform(input_df[SCALE_COLS])

            pred_class = int(model.predict(input_scaled)[0])

            time.sleep(0.4)  # brief pause so the spinner is visibly noticeable

        st.subheader(f"Predicted status: {CLASS_LABELS[pred_class]}")
        st.toast("New result ready", icon="✅")
