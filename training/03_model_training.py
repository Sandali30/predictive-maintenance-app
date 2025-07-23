# notebooks/03_model_training.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Load preprocessed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_score = 0
best_model_name = None
best_model = None

print("🔍 Training and evaluating models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ {name} Accuracy: {acc:.4f}")
    
    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_model = model

print("\n🏆 Best Model:", best_model_name)

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/final_model.pkl")
print("✅ Model saved as models/final_model.pkl")
