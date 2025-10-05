import pickle
import numpy as np
import streamlit as st

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Height Prediction App")

# Input widgets
weight = st.number_input("Weight")
age = st.number_input("Age")
shoe_size = st.number_input("Shoe Size")
arm_length = st.number_input("Arm Length")
leg_length = st.number_input("Leg Length")

if st.button("Predict"):
    features = np.array([[weight, age, shoe_size, arm_length, leg_length]])
    prediction = model.predict(features)
    st.write(f"Predicted Height: {round(prediction[0],2)} units")
