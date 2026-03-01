import streamlit as st
import pandas as pd
import os
import sys

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


# Page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)


# Custom CSS for better UI
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    height: 50px;
    width: 100%;
    border-radius: 10px;
}

.stNumberInput input {
    background-color: #262730;
    color: white;
}

.title {
    text-align: center;
    color: #FF4B4B;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)


# Title
st.markdown('<p class="title">Credit Risk Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict whether a person is High Risk or Low Risk</p>', unsafe_allow_html=True)


# Input Form
with st.container():

    col1, col2 = st.columns(2)

    with col1:
        util = st.number_input("Revolving Utilization", min_value=0.0, step=0.01)
        age = st.number_input("Age", min_value=18.0, step=1.0)
        past3059 = st.number_input("30-59 Days Past Due", min_value=0.0, step=1.0)
        debt = st.number_input("Debt Ratio", min_value=0.0, step=0.01)
        income = st.number_input("Monthly Income", min_value=0.0, step=100.0)

    with col2:
        opencredit = st.number_input("Open Credit Lines", min_value=0.0, step=1.0)
        late90 = st.number_input("90 Days Late", min_value=0.0, step=1.0)
        realestate = st.number_input("Real Estate Loans", min_value=0.0, step=1.0)
        late6089 = st.number_input("60-89 Days Past Due", min_value=0.0, step=1.0)
        dependents = st.number_input("Dependents", min_value=0.0, step=1.0)


# Predict button
if st.button("Predict Risk"):

    try:

        data = CustomData(
            util,
            age,
            past3059,
            debt,
            income,
            opencredit,
            late90,
            realestate,
            late6089,
            dependents
        )

        df = data.get_data_as_dataframe()

        pipeline = PredictPipeline()

        prediction = pipeline.predict(df)

        if prediction[0] == 1:
            st.error("HIGH RISK CUSTOMER")
        else:
            st.success("LOW RISK CUSTOMER")

    except Exception as e:
        st.error(f"Error: {e}")


# Footer
st.markdown("---")
st.markdown(
    "<center>Built with Streamlit | ML Credit Risk Project</center>",
    unsafe_allow_html=True
)
