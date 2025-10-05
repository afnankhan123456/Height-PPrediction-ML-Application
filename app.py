import pickle
import numpy as np
import streamlit as st
import requests
import io
import sys

st.set_page_config(page_title="Height Prediction App")
st.title("Height Prediction App")

# GitHub raw link
url = "https://github.com/afnankhan123456/Height-PPrediction-ML-Application/raw/main/model.pkl"

# Download and load model with error handling
try:
    response = requests.get(url)
    response.raise_for_status()  # Raise error for bad HTTP response
    model = pickle.load(io.BytesIO(response.content))
except requests.exceptions.RequestException as e:
    st.error(f"Failed to download model: {e}")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"Module not found while loading model: {e}. Make sure all required packages are in requirements.txt")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input widgets
weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
age = st.number_input("Age (years)", min_value=0, step=1)
shoe_size = st.number_input("Shoe Size", min_value=0.0, step=0.1)
arm_length = st.number_input("Arm Length (cm)", min_value=0.0, step=0.1)
leg_length = st.number_input("Leg Length (cm)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict Height"):
    try:
        features = np.array([[weight, age, shoe_size, arm_length, leg_length]])
        prediction = model.predict(features)
        st.success(f"Predicted Height: {round(prediction[0],2)} units")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
