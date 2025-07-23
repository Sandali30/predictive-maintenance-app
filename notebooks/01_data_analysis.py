# notebooks/01_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output directory
os.makedirs("notebooks/output", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/ai4i2020.csv")

# Basic information
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Null Values ---")
print(df.isnull().sum())

# Check target column distribution
print("\n--- Machine Failure Distribution ---")
print(df['Machine failure'].value_counts())

# Plot distribution of numerical features
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove("Machine failure")  # Exclude target

plt.figure(figsize=(15, 10))
df[numeric_cols].hist(bins=30, figsize=(15, 10), layout=(4, 3))
plt.tight_layout()
plt.savefig("notebooks/output/numeric_distributions.png")
plt.close()

# ✅ Add this section to create and analyze "Failure Type" column
def identify_failure_type(row):
    if row['TWF'] == 1:
        return 'Tool Wear Failure'
    elif row['HDF'] == 1:
        return 'Heat Dissipation Failure'
    elif row['PWF'] == 1:
        return 'Power Failure'
    elif row['OSF'] == 1:
        return 'Overstrain Failure'
    elif row['RNF'] == 1:
        return 'Random Failures'
    else:
        return 'No Failure'

df['Failure Type'] = df.apply(identify_failure_type, axis=1)

# Print distribution
print("\n--- Failure Type Distribution ---")
print(df['Failure Type'].value_counts())

# Optional: Visualize Failure Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Failure Type', order=df['Failure Type'].value_counts().index, palette='Set2')
plt.title("Failure Type Distribution")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("notebooks/output/failure_type_distribution.png")
plt.close()
