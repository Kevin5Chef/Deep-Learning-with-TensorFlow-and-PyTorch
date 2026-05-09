import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
print("SY-5, Kevin Victor, Roll No.-30")
# =========================================================
# 1. SIMULATED MULTI-SCALE HEATWAVE DATASET
# =========================================================

def generate_dataset(n_samples=1000):
    np.random.seed(42)

    data = pd.DataFrame({

        # -------------------------------
        # LOCAL FEATURES
        # -------------------------------
        "temp": np.random.uniform(25, 48, n_samples),
        "humidity": np.random.uniform(20, 90, n_samples),
        "pressure": np.random.uniform(990, 1025, n_samples),
        "wind_speed": np.random.uniform(0, 15, n_samples),

        # -------------------------------
        # MID-LEVEL FEATURES
        # -------------------------------
        "wind_convergence": np.random.uniform(-5, 5, n_samples),
        "vertical_velocity": np.random.uniform(-2, 2, n_samples),
        "geopotential_height_500": np.random.uniform(5400, 5900, n_samples),
        "mixing_layer_height": np.random.uniform(500, 3000, n_samples),

        # -------------------------------
        # GLOBAL FEATURES
        # -------------------------------
        "enso_index": np.random.uniform(-2.5, 2.5, n_samples),  # El Niño / La Niña
        "iod_index": np.random.uniform(-1.5, 1.5, n_samples),
        "sst_anomaly": np.random.uniform(-3, 3, n_samples)
    })

    # Target: Heatwave (1 or 0)
    # Simple synthetic rule (non-linear reality simplified)
    data["heatwave"] = (
        (data["temp"] > 40) &
        (data["humidity"] < 50) &
        (data["enso_index"] > 0.5)
    ).astype(int)

    return data


# =========================================================
# 2. FEATURE ENGINEERING (CUSTOM TRANSFORM)
# =========================================================

def feature_engineering(df):
    df = df.copy()

    # Heat Index (approx proxy)
    df["heat_index"] = df["temp"] * df["humidity"] / 100

    # Pressure anomaly
    df["pressure_anomaly"] = df["pressure"] - df["pressure"].mean()

    # Atmospheric instability proxy
    df["instability"] = df["vertical_velocity"] * df["mixing_layer_height"]

    # ENSO lag-like interaction
    df["enso_sst_interaction"] = df["enso_index"] * df["sst_anomaly"]

    return df


# =========================================================
# 3. CUSTOM PYTORCH DATASET
# =========================================================

class HeatwaveDataset(Dataset):
    def __init__(self, dataframe):
        df = feature_engineering(dataframe)

        self.X = df.drop("heatwave", axis=1).values.astype(np.float32)
        self.y = df["heatwave"].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# =========================================================
# 4. DATA PREPROCESSING + DATALOADER
# =========================================================

data = generate_dataset(1000)

dataset = HeatwaveDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# =========================================================
# 5. MODEL BUILDING (SIMPLE LINEAR MODEL)
# =========================================================

class HeatwaveModel(nn.Module):
    def __init__(self, input_dim):
        super(HeatwaveModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Determine input size dynamically
sample_X, _ = dataset[0]
input_dim = sample_X.shape[0]

model = HeatwaveModel(input_dim)


# =========================================================
# 6. PRINT MODEL STRUCTURE
# =========================================================

print("\n===== MODEL STRUCTURE =====")
print(model)

# =========================================================
# 7. MODEL COMPILATION (PYTORCH STYLE)
# =========================================================

criterion = nn.BCEWithLogitsLoss()   # For binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================================================
# 8. DEBUGGING UTILITIES (PYTORCH STRATEGY)
# =========================================================

def debug_batch(x, y, outputs, loss, step):
    print(f"\n===== DEBUG STEP {step} =====")

    print("Input sample (first row):", x[0])
    print("Raw output (logit):", outputs[0].item())
    print("Actual label:", y[0].item())
    print("Loss:", loss.item())

    # Convert logits → probability
    prob = torch.sigmoid(outputs)
    print("Predicted probability:", prob[0].item())

    pred = (prob > 0.5).float()
    print("Predicted class:", pred[0].item())

    return prob, pred


def provenance_report(step, prev_loss, current_loss):
    if step == 1:
        return

    print("\n--- PROVENANCE REPORT ---")

    if current_loss < prev_loss:
        print(f"Step {step}: Loss decreased → Model is learning meaningful patterns.")
        print("Reason: Weight updates improved alignment with heatwave conditions.")
    else:
        print(f"Step {step}: Loss increased → Model struggling or noisy gradients.")
        print("Reason: Possible imbalance / learning rate / feature scaling issue.")

    print("Improvement focus:")
    print("- Better weight tuning via backprop")
    print("- Gradual capture of multi-scale climate interactions")


# =========================================================
# 9. TRAINING LOOP (WITH INTERMEDIATE DEBUGGING)
# =========================================================

epochs = 3
debug_steps = 6
step_counter = 0
prev_loss = float('inf')

for epoch in range(epochs):
    print(f"\n===== EPOCH {epoch+1} =====")

    for X_batch, y_batch in dataloader:

        # Forward pass
        outputs = model(X_batch).squeeze()

        # Loss
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Debug (important PyTorch trick)
        if step_counter < debug_steps:
            print("\nGradient (first weight):", list(model.parameters())[0].grad[0][0].item())

        optimizer.step()

        # Debugging outputs (limit to 6)
        if step_counter < debug_steps:
            prob, pred = debug_batch(X_batch, y_batch, outputs, loss, step_counter + 1)

            provenance_report(step_counter + 1, prev_loss, loss.item())

            prev_loss = loss.item()
            step_counter += 1

        if step_counter >= debug_steps:
            break

    if step_counter >= debug_steps:
        break


# =========================================================
# 10. EVALUATION (MANUAL)
# =========================================================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in dataloader:
        outputs = model(X_batch).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total

print("\n===== EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")


# =========================================================
# 11. VALIDATION INSIGHT (MANUAL ANALYSIS)
# =========================================================

print("\n===== VALIDATION INSIGHT =====")

print("""
Observations:
- Model is learning from synthetic climate signals
- Linear model may struggle with complex atmospheric interactions

Potential Improvements:
- Add non-linearity (ReLU layers)
- Use time-series models (LSTM for climate evolution)
- Normalize features (very important for stability)
- Add class balancing (heatwaves are rare events)

Debug Strategy Used (PyTorch-specific):
- Manual forward inspection
- Gradient tracking
- Loss trend analysis
- Prediction vs actual comparison

This level of control is NOT directly available in TensorFlow's .fit()
and is why PyTorch excels in research and debugging scenarios.
""")