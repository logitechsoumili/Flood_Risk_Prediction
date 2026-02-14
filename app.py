import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide"
)

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

                st.divider()
                st.subheader("Risk Assessment Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Predicted Risk Level", risk_labels[risk_level])

                with col2:
                    st.metric("Model Confidence", f"{np.max(probabilities) * 100:.2f}%")

                with col3:
                    st.metric("Elevation (m)", f"{elevation:.2f}")

                st.divider()

                st.subheader("Live Weather Data")

                st.write(f"Rainfall (1h): {rainfall} mm")
                st.write(f"Temperature: {temperature} °C")
                st.write(f"Humidity: {humidity} %")


        # Probability Distribution Chart
        st.subheader("Risk Probability Distribution")

        fig, ax = plt.subplots()

        bars = ax.bar(risk_labels, probabilities)

        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title("Predicted Risk Probabilities")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)

        st.divider()

        st.markdown(
            """
            **Risk Level Interpretation**

            - **Low:** Minimal hydrological stress conditions  
            - **Medium:** Moderate flood potential  
            - **High:** Elevated flood likelihood; monitoring recommended  
            """
        )

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