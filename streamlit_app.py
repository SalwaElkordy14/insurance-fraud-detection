import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Insurance Fraud Detector", page_icon="🛡️", layout="wide")

with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title("🛡️ Insurance Fraud Detection")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Details")
    insured_age = st.slider("Age", 18, 80, 30)
    insured_sex = st.selectbox("Sex", ["Male", "Female"])
    insured_education_level = st.selectbox("Education", ["High School", "College", "Masters", "PhD"])
    insured_occupation = st.selectbox("Occupation", ["craft-repair", "machine-op-inspct", "sales", "armed-forces", "tech-support", "other-service"])
    insured_hobbies = st.selectbox("Hobbies", ["sleeping", "reading", "board-games", "base-jumping", "bungie-jumping", "other"])

with col2:
    st.subheader("🚗 Incident Details")
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"])
    collision_type = st.selectbox("Collision Type", ["Front Collision", "Rear Collision", "Side Collision", "Unknown"])
    incident_severity = st.selectbox("Severity", ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
    authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "Unknown"])
    incident_hour_of_the_day = st.slider("Incident Hour", 0, 23, 12)

with col3:
    st.subheader("📋 Policy Details")
    policy_deductible = st.selectbox("Deductible", [500, 1000, 2000])
    policy_annual_premium = st.number_input("Annual Premium", 500, 3000, 1000)
    number_of_vehicles_involved = st.slider("Vehicles Involved", 1, 4, 1)
    bodily_injuries = st.slider("Bodily Injuries", 0, 2, 0)
    witnesses = st.slider("Witnesses", 0, 3, 0)
    police_report_available = st.selectbox("Police Report", ["Yes", "No"])
    claim_amount = st.number_input("Claim Amount", 0, 100000, 10000)
    total_claim_amount = st.number_input("Total Claim Amount", 0, 100000, 15000)

st.markdown("---")

if st.button("🔍 Detect Fraud", use_container_width=True):
    input_dict = {col: 0 for col in feature_names}
    
    input_dict['insured_age'] = insured_age
    input_dict['policy_deductible'] = policy_deductible
    input_dict['policy_annual_premium'] = policy_annual_premium
    input_dict['incident_hour_of_the_day'] = incident_hour_of_the_day
    input_dict['number_of_vehicles_involved'] = number_of_vehicles_involved
    input_dict['bodily_injuries'] = bodily_injuries
    input_dict['witnesses'] = witnesses
    input_dict['claim_amount'] = claim_amount
    input_dict['total_claim_amount'] = total_claim_amount

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ FRAUD DETECTED! Probability: {prob:.2%}")
    else:
        st.success(f"✅ LEGITIMATE CLAIM! Fraud Probability: {prob:.2%}")