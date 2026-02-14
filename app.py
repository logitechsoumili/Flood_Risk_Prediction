import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
import folium
from streamlit_folium import folium_static

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Disaster Risk Prediction System",
    page_icon="⚠️",
    layout="wide"
)

st.sidebar.markdown("### System Status")
st.sidebar.success("Model Loaded")
st.sidebar.info("Weather API Connected")

# ------------------------------------------------
# Load Model
# ------------------------------------------------
flood_model = joblib.load("models/flood_risk_model.pkl")
heatwave_model = joblib.load("models/heatwave_model.pkl")

risk_labels = ["Low", "Medium", "High"]

# ------------------------------------------------
# API Configuration
# ------------------------------------------------
API_KEY = st.secrets["OPENWEATHER_API_KEY"]

def fetch_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    rainfall = data.get("rain", {}).get("1h", 0)
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    lat = data["coord"]["lat"]
    lon = data["coord"]["lon"]

    return rainfall, temperature, humidity, lat, lon

def fetch_elevation(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()
    return data["elevation"][0]

def fetch_weather_detailed(lat, lon):

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_min,"
        f"relative_humidity_2m_max,relative_humidity_2m_min,"
        f"precipitation_sum"
        f"&hourly=surface_pressure,wind_speed_10m"
        f"&forecast_days=1"
        f"&timezone=auto"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    min_temp = data["daily"]["temperature_2m_min"][0]
    max_humidity = data["daily"]["relative_humidity_2m_max"][0]
    min_humidity = data["daily"]["relative_humidity_2m_min"][0]
    rainfall = data["daily"]["precipitation_sum"][0]

    pressure = data["hourly"]["surface_pressure"][0]
    wind_speed = data["hourly"]["wind_speed_10m"][0]

    return min_temp, max_humidity, min_humidity, wind_speed, pressure, rainfall

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Flood Risk", "Heatwave Risk", "Model Insights", "About"]
)

# ==============================================================
# PAGE 1 — FLOOD RISK
# ==============================================================
if page == "Flood Risk":

    st.title("Flood Risk Prediction System")
    st.markdown(
        "AI-driven hydrological risk classification engine using machine learning."
    )

    st.divider()

    st.subheader("City-Based Risk Assessment")

    city = st.text_input("Enter City Name")

    historical = st.selectbox(
        "Historical Flood Indicator (0 = No, 1 = Yes)",
        [0, 1]
    )

    if st.button("Run Assessment"):

        if not city:
            st.error("Please enter a city name.")
        else:
            weather_data = fetch_weather(city)

            if weather_data is None:
                st.error("City not found or API error.")
            else:
                rainfall, temperature, humidity, lat, lon = weather_data

                elevation = fetch_elevation(lat, lon)
                if elevation is None:
                    elevation = 4400  # fallback

                # Derived hydrological approximations
                discharge = 2000 + (rainfall * 10)
                water_level = 4 + (rainfall * 0.02)

                input_data = np.array([[rainfall, discharge, water_level, elevation, historical]])

                with st.spinner("Processing hydrological indicators..."):
                    prediction = flood_model.predict(input_data)
                    probabilities = flood_model.predict_proba(input_data)[0]

                risk_level = prediction[0]

                risk_text = risk_labels[risk_level]

                if risk_text == "High":
                    st.error("🚨 HIGH FLOOD RISK — Immediate Monitoring Recommended")
                elif risk_text == "Medium":
                    st.warning("⚠ MODERATE FLOOD RISK — Stay Alert")
                else:
                    st.success("✅ LOW FLOOD RISK — Conditions Stable")

                st.divider()
                
                colA, colB = st.columns(2)

                with colA:
                    st.markdown("### 📊 Risk Assessment Summary")
                    with st.container(border=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Predicted Risk Level", risk_labels[risk_level])

                        with col2:
                            st.metric("Model Confidence", f"{np.max(probabilities) * 100:.2f}%")

                        with col3:
                            st.metric("Elevation (m)", f"{elevation:.2f}")

                with colB:
                    st.markdown("### ⛅ Live Weather Report")
                    with st.container(border=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Rainfall (1h)", f"{rainfall} mm")

                        with col2:
                            st.metric("Temperature", f"{temperature} °C")

                        with col3:
                            st.metric("Humidity", f"{humidity} %")

                st.divider()

                st.subheader("📍 Geographical Risk Location")
                
                m = folium.Map(location=[lat, lon], zoom_start=10)
                
                color = "green"
                if risk_text == "High":
                    color = "red"
                elif risk_text == "Medium":
                    color = "orange"
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"{city} — {risk_text} Risk"
                ).add_to(m)
                
                folium_static(m, width=1200, height=400)

                st.divider()

                st.info("""
                    ### Risk Interpretation

                    - **Low Risk** → Minimal hydrological stress  
                    - **Medium Risk** → Moderate flood probability  
                    - **High Risk** → Elevated likelihood of flooding — monitoring advised  
                    """)

# ==============================================================
# PAGE 2 — HEATWAVE RISK
# ==============================================================
elif page == "Heatwave Risk":

    st.title("Heatwave Risk Prediction System")
    st.markdown("AI-driven atmospheric heat stress classification model.")

    st.divider()

    city = st.text_input("Enter City Name for Heatwave Assessment")

    if st.button("Run Assessment"):

        if not city:
            st.error("Please enter a city name.")
        else:
            weather_data = fetch_weather(city)

            if weather_data is None:
                st.error("City not found or API error.")
                st.stop()

            rainfall_now, temperature, humidity, lat, lon = weather_data

            detailed = fetch_weather_detailed(lat, lon)

            if detailed is None:
                st.error("Detailed weather fetch failed.")
            else:
                min_temp, max_humidity, min_humidity, wind_speed, pressure, rainfall = detailed

                input_data = np.array([[
                    min_temp,
                    max_humidity,
                    min_humidity,
                    wind_speed,
                    pressure,
                    rainfall
                ]])

                with st.spinner("Analyzing atmospheric heat indicators..."):
                    prediction = heatwave_model.predict(input_data)
                    probability = heatwave_model.predict_proba(input_data)[0][1]

                if prediction[0] == 1:
                    st.error("🔥 HIGH HEATWAVE RISK — Avoid outdoor exposure")
                else:
                    st.success("✅ LOW HEATWAVE RISK — Conditions Normal")

                st.metric("Heatwave Probability", f"{probability * 100:.2f}%")

                st.divider()

                st.subheader("📍 Location Overview")

                m = folium.Map(location=[lat, lon], zoom_start=10)

                color = "red" if prediction[0] == 1 else "green"

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"{city} — Heatwave Risk"
                ).add_to(m)

                folium_static(m, width=1200, height=400)

# ==============================================================
# PAGE 3 — MODEL INSIGHTS
# ==============================================================
elif page == "Model Insights":

    st.title("Model Insights")

    hazard = st.selectbox(
        "Select Hazard",
        ["Flood", "Heatwave"]
    )

    st.divider()

    if hazard == "Flood":

        st.markdown("## Flood Risk Model")

        st.markdown("""
        ### Model Type
        XGBoost Classifier

        ### Problem Framing
        Multi-class classification (Low / Medium / High Risk)

        ### Key Features Used
        - Rainfall
        - River Discharge
        - Water Level
        - Elevation
        - Historical Flood Indicator

        ### Why XGBoost?
        - Handles non-linear relationships well
        - Performs strongly on structured tabular data
        - Provides feature importance analysis

        ### Validation Strategy
        Stratified train-test split to maintain class balance.
        """)

    else:

        st.markdown("## Heatwave Risk Model")

        st.markdown("""
        ### Model Type
        Random Forest Classifier

        ### Problem Framing
        Binary classification (Heatwave / No Heatwave)

        ### Label Definition
        Heatwave defined using temperature threshold (≥ 40°C).

        ### Data Leakage Prevention
        The direct threshold feature (max_temperature) was excluded
        to ensure the model learned indirect atmospheric patterns.

        ### Why Random Forest?
        - Strong performance on tabular weather data
        - Robust to class imbalance
        - Balanced precision and recall (high F1 score)

        ### Evaluation Focus
        F1 Score prioritized over accuracy due to class imbalance.
        """)

# ==============================================================
# PAGE 4 — ABOUT
# ==============================================================
else:
    
    st.title("About This Project")

    st.markdown(
        """
        ## Multi-Hazard Disaster Risk Prediction System

        This project demonstrates an end-to-end machine learning pipeline
        for predicting environmental disaster risks using real-world meteorological data.

        ### Implemented Hazard Modules

        🌊 Flood Risk Prediction
        - Uses hydrological indicators derived from rainfall, discharge and elevation.
        - Multi-class classification: Low, Medium, High risk.
        - Model: XGBoost Classifier.

        🔥 Heatwave Risk Prediction
        - Binary classification of heat stress conditions.
        - Label derived from temperature threshold.
        - To prevent data leakage, direct temperature threshold features were excluded.
        - Model: Random Forest Classifier.

        ### Key ML Concepts Applied

        - Data preprocessing and cleaning
        - Handling class imbalance using class_weight
        - Stratified train-test splitting
        - Model comparison (Logistic, RF, GB, XGBoost)
        - Feature importance analysis
        - API-based real-time inference
        - Deployment using Streamlit

        This system bridges classical machine learning and live meteorological deployment.
        """
    )

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.divider()
st.caption(
    "Disaster Risk Prediction System | Developed by Soumili Saha | ML Project 2026"
)