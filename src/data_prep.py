import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW, OUT = "data/raw", "data/processed"

paths = sorted(glob.glob(os.path.join(RAW, "*.csv")))
if not paths:
    sys.exit("No CSVs in data/raw/ - copy the dataset files there first.")

frames = []
for p in paths:
    print("Loading", os.path.basename(p))
    frames.append(pd.read_csv(p, low_memory=False, encoding="latin1"))
df = pd.concat(frames, ignore_index=True)
print("Rows loaded:", len(df))

df.columns = [c.strip() for c in df.columns]           # fix ' Label' etc.
df = df.replace([np.inf, -np.inf], np.nan).dropna()     # Infinity / NaN rows
df = df.drop_duplicates().reset_index(drop=True)
df["Label"] = df["Label"].astype(str).str.strip()
print("Rows after cleaning:", len(df))

print("\nClass balance:")
print(df["Label"].value_counts().to_string())

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Label"])
os.makedirs(OUT, exist_ok=True)
train.to_parquet(f"{OUT}/train.parquet", index=False)
test.to_parquet(f"{OUT}/test.parquet", index=False)
print(f"\nSaved {len(train)} train rows and {len(test)} test rows to {OUT}/")
