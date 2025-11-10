import streamlit as st
import requests
import pandas as pd

st.title("Hospital Readmission Predictor")

st.write("Enter patient data to predict 30-day readmission")

# Input fields
age = st.number_input("Age", 18, 100, 65)
sex = st.selectbox("Sex", ["M", "F"])
insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Private", "Uninsured"])
length_of_stay = st.number_input("Length of Stay", 1, 100, 5)
num_prior_admissions_6mo = st.number_input("Prior Admissions (6mo)", 0, 20, 1)
num_ed_visits_30d = st.number_input("ED Visits (30d)", 0, 10, 0)
num_medications = st.number_input("Number of Medications", 0, 50, 8)
hr_diagnosis_flag = st.selectbox("High-risk Diagnosis Flag", [0,1])
last_cr = st.number_input("Last Creatinine", 0.1, 10.0, 1.1)
social_risk_flag = st.selectbox("Social Risk Flag", [0,1])
discharge_dest = st.selectbox("Discharge Destination", ["Home","SNF","Rehab","HomeWithCare"])

if st.button("Predict"):
    payload = {
        "age": age,
        "sex": sex,
        "insurance": insurance,
        "length_of_stay": length_of_stay,
        "num_prior_admissions_6mo": num_prior_admissions_6mo,
        "num_ed_visits_30d": num_ed_visits_30d,
        "num_medications": num_medications,
        "hr_diagnosis_flag": hr_diagnosis_flag,
        "last_cr": last_cr,
        "social_risk_flag": social_risk_flag,
        "discharge_dest": discharge_dest
    }
    response = requests.post("http://127.0.0.1:5000/predict", json=payload)
    if response.status_code == 200:
        st.write(response.json())
    else:
        st.error("Error in prediction")
