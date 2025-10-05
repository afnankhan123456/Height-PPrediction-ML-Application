import pickle
import numpy as np
import streamlit as st
import requests
import io

st.set_page_config(page_title="Height Prediction App")
st.title("📏 Height Prediction App")

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

# Input widgets with big labels and unique keys
st.markdown("<h2 style='font-size:30px;'>⚖️ Weight (kg)</h2>", unsafe_allow_html=True)
weight = st.number_input("Weight", min_value=0.0, step=0.1, key="weight")

st.markdown("<h2 style='font-size:30px;'>🎂 Age (years)</h2>", unsafe_allow_html=True)
age = st.number_input("Age", min_value=0, step=1, key="age")

st.markdown("<h2 style='font-size:30px;'>👟 Shoe Size</h2>", unsafe_allow_html=True)
shoe_size = st.number_input("Shoe Size", min_value=0.0, step=0.1, key="shoe_size")

st.markdown("<h2 style='font-size:30px;'>💪 Arm Length (cm)</h2>", unsafe_allow_html=True)
arm_length = st.number_input("Arm Length", min_value=0.0, step=0.1, key="arm_length")

st.markdown("<h2 style='font-size:30px;'>🦵 Leg Length (cm)</h2>", unsafe_allow_html=True)
leg_length = st.number_input("Leg Length", min_value=0.0, step=0.1, key="leg_length")

# Prediction button
if st.button("Predict Height"):
    try:
        features = np.array([[weight, age, shoe_size, arm_length, leg_length]])
        prediction = model.predict(features)
        st.markdown(f"<h1 style='font-size:50px;'>📏 Predicted Height: {round(prediction[0],2)} cm</h1>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
