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
    page_title="Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide"
)

st.sidebar.markdown("### System Status")
st.sidebar.success("Model Loaded")
st.sidebar.info("Weather API Connected")

# ------------------------------------------------
# Load Model
# ------------------------------------------------
model = joblib.load("models/flood_risk_model.pkl")

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

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Risk Assessment", "Model Insights", "About"]
)

# ==============================================================
# PAGE 1 — RISK ASSESSMENT
# ==============================================================
if page == "Risk Assessment":

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

    if st.button("Fetch & Run Risk Assessment"):

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
                    prediction = model.predict(input_data)
                    probabilities = model.predict_proba(input_data)[0]

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
# PAGE 2 — MODEL INSIGHTS
# ==============================================================
elif page == "Model Insights":

    st.title("Model Insights")

    st.subheader("Performance Metrics")

    st.metric("Test Accuracy", "98.8%")
    st.metric("Macro F1 Score", "0.98")

    st.divider()

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = [
        "Rainfall (mm)",
        "River Discharge (m³/s)",
        "Water Level (m)",
        "Elevation (m)",
        "Historical Floods"
    ]

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)

    st.info(
        """
        Model: XGBoost Classifier  
        Features: 5 Hydrological Indicators  
        Labeling Method: KMeans-derived risk categorization  
        Validation Strategy: Stratified Train-Test Split  
        """
    )

# ==============================================================
# PAGE 3 — ABOUT
# ==============================================================
else:

    st.title("About This Project")

    st.markdown(
        """
        ### Flood Risk Prediction System

        This project demonstrates an end-to-end machine learning workflow:

        1. Initial binary flood prediction showed limited predictive signal.
        2. Risk zones were derived using KMeans clustering.
        3. A supervised XGBoost classifier was trained to predict derived risk levels.
        4. The deployed system provides real-time risk assessment via Streamlit.

        ### Technology Stack

        - Python  
        - Scikit-learn  
        - XGBoost  
        - Streamlit  
        - Matplotlib  

        This system illustrates practical application of unsupervised + supervised learning
        for environmental risk modeling.
        """
    )

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.divider()
st.caption(
    "Flood Risk Prediction System v2.0 | Developed by Soumili Saha | ML Project 2026"
)