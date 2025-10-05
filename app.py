import pickle
import numpy as np
import streamlit as st
import requests
import io

st.title("Height Prediction App")

# GitHub raw link
url = "https://github.com/afnankhan123456/Height-PPrediction-ML-Application/raw/main/model.pkl"

# Download model
response = requests.get(url)
if response.status_code == 200:
    model = pickle.load(io.BytesIO(response.content))
else:
    st.error("Failed to download model.")
    st.stop()

# Input widgets
weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
age = st.number_input("Age (years)", min_value=0, step=1)
shoe_size = st.number_input("Shoe Size", min_value=0.0, step=0.1)
arm_length = st.number_input("Arm Length (cm)", min_value=0.0, step=0.1)
leg_length = st.number_input("Leg Length (cm)", min_value=0.0, step=0.1)

if st.button("Predict Height"):
    features = np.array([[weight, age, shoe_size, arm_length, leg_length]])
    prediction = model.predict(features)
    st.success(f"Predicted Height: {round(prediction[0],2)} units")
