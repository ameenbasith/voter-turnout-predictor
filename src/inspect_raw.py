import pandas as pd
import os

raw_dir = "data/raw"
files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

for file in files:
    print(f"\n===== {file} =====")
    path = os.path.join(raw_dir, file)
    try:
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Dtypes:\n", df.dtypes)
        print("\nMissing values (top 10):")
        print(df.isnull().sum().sort_values(ascending=False).head(10))
        print("\nSample rows:")
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {file}: {e}")
