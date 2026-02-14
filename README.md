# 🌊 Flood Risk Prediction System

An end-to-end machine learning system for assessing flood risk using hydrological indicators and advanced machine learning techniques.

---

## 📷 Application Preview

### 🔹 Risk Assessment Dashboard

![Risk Dashboard](assets/dashboard.png)

---

### 🔹 Model Insights & Feature Importance

![Model Insights](assets/model_insights.png)

---

## 📌 Problem Statement

India experiences frequent natural disasters, with floods being one of the most recurring and destructive hazards. Rapid urbanization, climate variability, and changing rainfall patterns increase the vulnerability of many regions to flooding.

Early risk assessment based on environmental indicators can help in:

- Proactive disaster preparedness  
- Infrastructure planning  
- Risk-aware decision making  
- Resource allocation during extreme weather events  

This project aims to develop a machine learning-based system that predicts **flood risk levels** using key hydrological and environmental indicators such as rainfall, river discharge, water level, elevation, and historical flood occurrence.

---

## 🎯 Project Objective

To design and deploy a flood risk prediction system that:

- Identifies hydrological stress patterns
- Classifies regions into **Low, Medium, and High risk**
- Provides real-time risk assessment through a web interface
- Demonstrates practical application of ML in environmental risk modeling

---

## 🧠 Methodology

### 1️⃣ Data Preparation
- Train-test split performed before clustering
- StandardScaler applied for clustering
- KMeans fitted only on training data to prevent leakage

### 2️⃣ Risk Category Derivation (Unsupervised Learning)
- KMeans clustering applied on hydrological features
- Clusters interpreted based on:
  - Water Level
  - River Discharge
  - Elevation
  - Historical Flood Presence
- Risk levels labeled as **Low, Medium, High**

### 3️⃣ Supervised Classification
- XGBoost multi-class classifier trained on derived risk labels
- Model learns nonlinear hydrological boundaries
- Final model achieved **~98% test accuracy**

---

## 📊 Model Performance

- Test Accuracy: **98.8%**
- Macro F1 Score: **0.98**
- Strong generalization across stratified test split

Feature importance analysis shows that:
- Historical Flood Presence
- Water Level
- River Discharge  

are the most influential factors in classification.

---

## ⚠ Challenges Faced

During initial experimentation, binary flood occurrence prediction yielded approximately 50% accuracy, indicating limited predictive signal in the synthetic dataset.

To address this:

- Risk categories were derived using unsupervised clustering
- The problem was reframed as risk classification instead of direct flood occurrence prediction
- This approach ensured structured and interpretable label formation

This pivot improved model reliability and interpretability.

---

## 💻 Deployment

The system is deployed using **Streamlit** and includes:

- Risk assessment dashboard
- KPI metrics (Predicted Risk, Confidence)
- Risk probability visualization
- Model insights section
- Corporate-style interface

---

## 🛠 Technology Stack

- Python  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Matplotlib  
- Pandas  
- NumPy  

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
````

---

## 📈 Future Enhancements

* Multi-hazard expansion (Cyclone / Heatwave modules)
* Integration with real-world meteorological datasets
* Geospatial risk heatmap visualization
* API-based deployment