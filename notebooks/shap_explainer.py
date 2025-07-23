import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load data and model
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
model = joblib.load("models/final_model.pkl")

# Create SHAP explainer
explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_test)

# Save summary plot (global feature importance)
os.makedirs("outputs/shap", exist_ok=True)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/shap/summary_plot.png")
print("✅ SHAP summary plot saved at outputs/shap/summary_plot.png")

# Save force plot for first prediction (local explanation)
force_plot_path = "outputs/shap/force_plot.html"
shap.save_html(force_plot_path, shap.plots.force(shap_values[0]))
print(f"✅ SHAP force plot for one prediction saved at {force_plot_path}")
