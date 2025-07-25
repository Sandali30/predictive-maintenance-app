# 🔧 AI-Powered Industrial Predictive Maintenance System

This project uses machine learning to predict failures in industrial equipment using the AI4I 2020 Predictive Maintenance Dataset. The goal is to help industries reduce downtime and perform cost-efficient maintenance by identifying potential failures in advance.

---

## 📊 Project Features

- Supervised machine learning for failure prediction
- SHAP-based interpretability for model transparency
- Streamlit dashboard for real-time prediction and visualization
- Clean modular code structure for reproducibility

---

## 🧠 ML Techniques Used

- Random Forest, XGBoost, Logistic Regression
- Data preprocessing and feature engineering
- Model evaluation: confusion matrix, precision, recall, F1-score
- SHAP for model explanation

---

## 🗂️ Folder Structure

industrial_predictive_maintenance/
│
├── data/
│ ├── raw/ # Original downloaded data
│ ├── processed/ # Cleaned and transformed data
│ └── external/ # Any external sources (optional)
│
├── notebooks/
│ ├── 01_data_analysis.ipynb # Exploratory Data Analysis
│ ├── 02_model_training.ipynb # Model training and tuning
│
├── src/
│ ├── data_preprocessing.py # Functions to load and clean data
│ ├── feature_engineering.py # Transformations and encoding
│ ├── train_model.py # Train and save ML models
│ ├── evaluate.py # Performance metrics and visualizations
│ ├── explain_model.py # SHAP model interpretability
│ └── predict.py # Load model and make predictions
│
├── models/
│ └── final_model.pkl # Trained model (joblib or pickle)
│
├── streamlit_app/
│ ├── app.py # Streamlit dashboard
│ ├── ui_components.py # Charts, layout, etc.
│ └── utils.py # Helper functions
│
├── reports/
│ ├── figures/ # Saved visualizations
│ └── model_performance.md # Model evaluation summary
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignore logs, models, venv, etc.


---

## 📁 Dataset

**AI4I 2020 Predictive Maintenance Dataset**  
Source: [Kaggle](https://www.kaggle.com/datasets/shubhendra7/ai4i2020-predictive-maintenance-dataset)

It contains sensor readings from manufacturing equipment along with failure labels, useful for predictive modeling.

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Sandali30/industrial-predictive-maintenance.git
cd industrial_predictive_maintenance

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt




