import streamlit as st
import pickle
import pandas as pd

# Load model and features
model = pickle.load(open("weather_model.pkl","rb"))
features = pickle.load(open("features.pkl","rb"))

st.title("🌤 Weather Temperature Prediction")

humidity = st.slider("Humidity",0.0,1.0,0.5)
wind_speed = st.number_input("Wind Speed (km/h)",0.0)
pressure = st.number_input("Pressure (millibars)",0.0)
visibility = st.number_input("Visibility (km)",0.0)
month = st.slider("Month",1,12,6)
day = st.slider("Day",1,31,15)
hour = st.slider("Hour",0,23,12)

if st.button("Predict Temperature"):

    input_data = pd.DataFrame([[0]*len(features)], columns=features)

    if "Humidity" in input_data.columns:
        input_data["Humidity"] = humidity

    if "Wind Speed (km/h)" in input_data.columns:
        input_data["Wind Speed (km/h)"] = wind_speed

    if "Pressure (millibars)" in input_data.columns:
        input_data["Pressure (millibars)"] = pressure

    if "Visibility (km)" in input_data.columns:
        input_data["Visibility (km)"] = visibility

    if "month" in input_data.columns:
        input_data["month"] = month

    if "day" in input_data.columns:
        input_data["day"] = day

    if "hour" in input_data.columns:
        input_data["hour"] = hour

    prediction = model.predict(input_data)

    st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")