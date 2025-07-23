# scripts/extract_dataset.py
import zipfile
import os

# Define paths
zip_path = "archive.zip"
extract_to = "data/raw"

# Make sure output directory exists
os.makedirs(extract_to, exist_ok=True)

# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Dataset extracted to:", extract_to)
