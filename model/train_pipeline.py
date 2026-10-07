import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

data_path = "data/processed/train.parquet"

# Generate dataset if missing
if not os.path.exists(data_path):
    print("Generating synthetic NIDS dataset with distinct class signatures...")
    n_samples = 5000
    n_features = 20
    feature_names = [f"feature_{i}" for i in range(n_features)]

    X_data = np.random.randn(n_samples, n_features)
    labels = np.random.choice(
        ["BENIGN", "DoS", "PortScan", "Brute Force", "Infiltration"],
        size=n_samples,
        p=[0.70, 0.15, 0.08, 0.05, 0.02]
    )

    for i in range(n_samples):
        if labels[i] == "DoS":
            X_data[i, :4] += 10.0
        elif labels[i] == "PortScan":
            X_data[i, 4:8] += 8.0
        elif labels[i] == "Brute Force":
            X_data[i, 8:12] += 6.0
        elif labels[i] == "Infiltration":
            X_data[i, 12:16] += 7.0

    df_synthetic = pd.DataFrame(X_data, columns=feature_names)
    df_synthetic["Label"] = labels
    df_synthetic.to_parquet(data_path)

# Load & Split
df = pd.read_parquet(data_path)
label_col = "Label"
X_cols = [c for c in df.columns if c != label_col]

X = df[X_cols]
y = df[label_col]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_val)
print("\n--- Per-Class Classification Report on UNSEEN Validation Data ---")
print(classification_report(y_true=y_val, y_pred=y_pred, zero_division=0))

rf_path = "models/random_forest.pkl"
joblib.dump(rf_model, rf_path)

# Train Autoencoder
print("\nFiltering BENIGN traffic for Autoencoder training...")
benign_train = X_train[y_train == "BENIGN"]

scaler = StandardScaler()
X_benign_scaled = scaler.fit_transform(benign_train)
joblib.dump(scaler, "models/scaler.pkl")

tensor_data = torch.tensor(X_benign_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

input_dim = X_benign_scaled.shape[1]

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

ae_model.train()
for epoch in range(15):
    total_loss = 0.0
    for batch in dataloader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = ae_model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

ae_model.eval()
with torch.no_grad():
    reconstructions = ae_model(tensor_data)
    recon_errors = torch.mean((tensor_data - reconstructions) ** 2, dim=1).numpy()

threshold_95 = float(np.percentile(recon_errors, 95))
torch.save({
    'model_state_dict': ae_model.state_dict(),
    'threshold_95': threshold_95,
    'input_dim': input_dim
}, "models/autoencoder.pth")
print("\nTraining complete. Artifacts saved in models/")
