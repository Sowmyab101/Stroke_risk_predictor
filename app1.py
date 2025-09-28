# app_light.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("stroke_model_light.keras")
scaler = joblib.load("scaler.save")

st.title("🧠 Stroke Risk Predictor")
st.write("Enter patient data to predict the likelihood of stroke.")

# Input widgets
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
ever_married = st.selectbox("Ever Married", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
#work_type = st.selectbox("Work Type", [0, 1, 2, 3, 4])

work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4}
work_type = work_type_map[work_type]

residence_type = st.selectbox("Residence Type", [0, 1], format_func=lambda x: "Rural" if x == 0 else "Urban")
avg_glucose_level = st.number_input("Avg Glucose Level", min_value=50.0, max_value=300.0, value=90.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
smoking_status = st.selectbox("Smoking Status", [0, 1, 2], format_func=lambda x: ["Unknown", "Never smoked", "Smokes"][x])

# Predict button
if st.button("Predict Stroke Risk"):
    input_data = np.array([[sex, age, hypertension, heart_disease, ever_married,
                            work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
    input_scaled = scaler.transform(input_data)
    
    # Fast prediction
    probability = model.predict(input_scaled, batch_size=1, verbose=0).squeeze()
    
    # Display risk
    st.write(f"### 🩺 Stroke Risk Probability: `{probability * 100:.2f}%`")
    if probability > 0.5:
        st.error("🚨 High risk of stroke")
    elif probability > 0.2:
        st.warning("⚠️ Moderate risk of stroke")
    else:
        st.success("✅ Low risk of stroke")
