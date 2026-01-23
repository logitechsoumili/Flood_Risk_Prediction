# 🌊 Flood Risk Prediction System

AI-powered environmental flood risk classification using machine learning and an interactive Gradio interface.

This project predicts flood risk levels (Low, Medium, High) based on environmental, hydrological, and socio-economic indicators.

---

## 🚀 Overview

Flood risk assessment is critical for disaster preparedness and mitigation.
This project implements an end-to-end machine learning pipeline:

* Risk label derivation using **unsupervised clustering**
* Multi-model comparison using supervised learning
* Model evaluation with accuracy, F1-score, and recall
* Deployment using an interactive **Gradio web interface**

The system classifies environmental conditions into:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Risk Derivation (Unsupervised Learning)

Since multi-class flood risk labels were not predefined in the dataset, risk levels were generated using **KMeans clustering** on key hydrological indicators:

* Rainfall
* River Discharge
* Water Level
* Elevation
* Historical Floods

Clusters were interpreted and labeled as Low, Medium, and High risk based on environmental characteristics.

---

### 2️⃣ Model Training & Comparison

The following models were evaluated:

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Evaluation Metrics:

* Accuracy
* Macro F1-score
* Macro Recall
* Confusion Matrix

Ensemble boosting models demonstrated superior performance, achieving:

* ~99% Accuracy
* Balanced Recall across all risk classes

---

## 🎨 Gradio Web Interface

The project includes a styled interactive UI built with **Gradio Blocks**, featuring:

* Slider-based environmental inputs
* Dynamic risk banner (color-coded)
* Probability breakdown display
* Clean two-column layout

The interface allows users to simulate environmental conditions and visualize predicted flood risk in real time.

---

## 📊 Example Output

```
🔴 HIGH RISK  
High flood risk detected. Immediate caution advised.

Probability Breakdown:
- High: 85.51%
- Low: 3.39%
- Medium: 1.10%
```

---

## 🗂️ Project Structure

```
├── model_training.ipynb
├── data
│   └── flood_dataset.csv
├── app.py
├── models/
│   ├── flood_risk_model.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/logitechsoumili/Flood_Risk_Prediction.git
cd flood-risk-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
python app.py
```

The app will launch locally at:

```
http://127.0.0.1:7860
```

---

## 📦 Dependencies

* Python 3.9+
* scikit-learn
* xgboost
* pandas
* numpy
* gradio
* joblib

---

## ⚠️ Note on Dataset

This project uses a synthetic flood risk dataset for educational and modeling purposes.
The predictions demonstrate machine learning workflow and system design rather than real-time hydrological forecasting.

---

## 🔮 Future Improvements

* Integrate real-time weather APIs
* Add elevation API support
* Deploy on Hugging Face Spaces
* Add map-based visualization
* Incorporate real historical flood datasets