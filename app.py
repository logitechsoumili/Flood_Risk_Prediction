import gradio as gr
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("models/flood_risk_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

def predict_flood(rainfall, temperature, humidity,
                  river_discharge, water_level, elevation,
                  population_density, infrastructure, historical_floods):

    input_dict = {
        "Rainfall (mm)": rainfall,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "River Discharge (m³/s)": river_discharge,
        "Water Level (m)": water_level,
        "Elevation (m)": elevation,
        "Population Density": population_density,
        "Infrastructure": infrastructure,
        "Historical Floods": historical_floods
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]

    risk_label = label_encoder.inverse_transform(prediction)[0]
    classes = label_encoder.classes_

    # Probability breakdown
    prob_output = "## 📊 Probability Breakdown\n\n" + "\n".join(
        [
            f"- **{classes[i]}**: {probabilities[i]*100:.2f}%"
            for i in range(len(classes))
        ]
    )

    if risk_label == "Low":
        banner = f"""
        <div style="background: linear-gradient(90deg,#0f5132,#198754);
                    padding:20px;
                    border-radius:12px;
                    color:white;">
        <h2>🟢 LOW RISK</h2>
        <p>Current environmental conditions indicate minimal flood risk.</p>
        </div>
        """
    elif risk_label == "Medium":
        banner = f"""
        <div style="background: linear-gradient(90deg,#664d03,#ffc107);
                    padding:20px;
                    border-radius:12px;
                    color:black;">
        <h2>🟡 MEDIUM RISK</h2>
        <p>Moderate flood likelihood. Monitoring recommended.</p>
        </div>
        """
    else:
        banner = f"""
        <div style="background: linear-gradient(90deg,#842029,#dc3545);
                    padding:20px;
                    border-radius:12px;
                    color:white;">
        <h2>🔴 HIGH RISK</h2>
        <p>High flood risk detected. Immediate caution advised.</p>
        </div>
        """

    return banner, prob_output


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
        # 🌊 Flood Risk Prediction System  
        ### AI-powered environmental risk classification
        """)

    with gr.Row():
        with gr.Column():

            rainfall = gr.Slider(0, 500, value=100, label="Rainfall (mm)")
            temperature = gr.Slider(-10, 50, value=25, label="Temperature (°C)")
            humidity = gr.Slider(0, 100, value=40, label="Humidity (%)")
            river_discharge = gr.Slider(0, 5000, value=500, label="River Discharge (m³/s)")
            water_level = gr.Slider(0, 6000, value=1000, label="Water Level (m)")
            elevation = gr.Slider(0, 5000, value=1000, label="Elevation (m)")
            population_density = gr.Slider(0, 500000, value=20000, label="Population Density")
            infrastructure = gr.Radio([0,1], value=1, label="Infrastructure (0=Poor, 1=Good)")
            historical_floods = gr.Radio([0,1], value=0, label="Historical Floods")

            submit = gr.Button("Predict Risk")

        with gr.Column():
            output_label = gr.Markdown()
            output_probs = gr.Markdown()

    submit.click(
        predict_flood,
        inputs=[rainfall, temperature, humidity,
                river_discharge, water_level, elevation,
                population_density, infrastructure, historical_floods],
        outputs=[output_label, output_probs]
    )

demo.launch()