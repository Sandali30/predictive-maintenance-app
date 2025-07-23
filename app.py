import streamlit as st
import streamlit_authenticator as stauth
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import base64
import matplotlib.pyplot as plt

# ---------- APP CONFIG ----------
st.set_page_config(page_title="Industrial Predictive Maintenance System", layout="wide")
st.title("🔧 Industrial Predictive Maintenance System")
st.write("Upload input data or simulate sensor input to predict machine failure.")

# ---------- LOAD MODEL ----------
model = joblib.load("models/final_model.pkl")

# ---------- LIVE SENSOR SIMULATION ----------
def simulate_live_input():
    types = ['H', 'L', 'M']
    data = {
        'Type': np.random.choice(types),
        'Air_temperature_K': np.random.normal(300, 5),
        'Process_temperature_K': np.random.normal(310, 5),
        'Rotational_speed_rpm': np.random.randint(1000, 3000),
        'Torque_Nm': np.random.uniform(10, 100),
        'Tool_wear_min': np.random.randint(0, 250)
    }
    return pd.DataFrame([data])

# ---------- PREDICTION ----------
def predict(df):
    try:
        df_encoded = pd.get_dummies(df)
        model_features = model.get_booster().feature_names
        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_features]
        prediction = model.predict(df_encoded)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# ---------- SHAP FEATURE IMPORTANCE ----------
def shap_summary_plot():
    st.subheader("🔍 Feature Importance (SHAP)")
    try:
        X_sample = pd.read_csv("data/X_sample.csv")
        X_encoded = pd.get_dummies(X_sample)
        model_features = model.get_booster().feature_names
        for col in model_features:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[model_features]

        explainer = shap.Explainer(model)
        shap_values = explainer(X_encoded)

        # SHAP plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, X_encoded, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP plot error: {e}")

# ---------- INPUT METHOD ----------
option = st.radio("Choose input method", ["Upload CSV", "Simulate Live Sensor Input"])

input_df = None
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload input CSV", type="csv")
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Input Data", input_df)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
elif option == "Simulate Live Sensor Input":
    if st.button("Generate Live Sensor Data"):
        input_df = simulate_live_input()
        st.write("### Simulated Sensor Input", input_df)

# ---------- PREDICTION OUTPUT ----------
if input_df is not None:
    predictions = predict(input_df)
    if predictions is not None:
        input_df['Prediction'] = predictions
        st.write("### 🔎 Prediction Results", input_df)

        # Download CSV
        csv = input_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">📥 Download Prediction Results</a>'
        st.markdown(href, unsafe_allow_html=True)

# ---------- SHAP CHECKBOX ----------
if st.checkbox("Show Feature Importance (SHAP)"):
    shap_summary_plot()
