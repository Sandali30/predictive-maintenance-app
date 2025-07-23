# notebooks/02_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
df = pd.read_csv("data/raw/ai4i2020.csv")

# Clean column names: remove brackets and replace spaces with underscores
df.columns = df.columns.str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")

# Drop unnecessary columns
df.drop(columns=["UDI", "Product_ID"], inplace=True)

# Encode 'Type' column (categorical)
le_type = LabelEncoder()
df["Type"] = le_type.fit_transform(df["Type"])

# Create 'Failure_Type' column from individual failure indicators
failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df['Failure_Type'] = df[failure_cols].idxmax(axis=1)
df.loc[df['Machine_failure'] == 0, 'Failure_Type'] = 'No Failure'

# Encode 'Failure_Type'
le_failure = LabelEncoder()
df["Failure_Type_Encoded"] = le_failure.fit_transform(df["Failure_Type"])

# Drop individual failure type indicator columns to avoid data leakage
df.drop(columns=failure_cols, inplace=True)

# Define features and target
X = df.drop(columns=["Machine_failure", "Failure_Type", "Failure_Type_Encoded"])
y = df["Machine_failure"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the processed datasets
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# Save the fully preprocessed and encoded DataFrame for reference
df.to_csv("data/processed/df_encoded.csv", index=False)

print("✅ Data preprocessing complete. Files saved in data/processed/")
