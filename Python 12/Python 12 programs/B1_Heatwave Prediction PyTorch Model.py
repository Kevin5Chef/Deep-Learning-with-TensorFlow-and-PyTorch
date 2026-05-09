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
# 7. EXPLANATION (PRINT STATEMENTS)
# =========================================================

print("\n===== EXPLANATION =====")

print("""
PyTorch Pipeline Demonstrated:

1. Data Preprocessing:
   - Custom dataset created using torch.utils.data.Dataset
   - DataLoader used for batching and shuffling

2. Feature Engineering:
   - Done manually using pandas/numpy
   - Includes:
        • Heat Index
        • Pressure anomaly
        • Atmospheric instability
        • ENSO-SST interaction

3. Model Building:
   - Defined using nn.Module
   - Single Linear Layer used

Key Insight:

Unlike TensorFlow:
- There is NO model.compile()
- There is NO automatic .fit()

Everything is explicitly controlled:
    • Data flow
    • Feature engineering
    • Training loop (not implemented here)

Why PyTorch is Powerful:

- Full control over every step
- Easy debugging (dynamic computation graph)
- Ideal for:
    • Research-level experimentation
    • Custom physics-based models
    • Hybrid AI systems (like climate + ML)

Real-world strategy:

- Use PyTorch for experimentation and model design
- Use TensorFlow for deployment pipelines (TF Serving, TFLite)

Conclusion:

PyTorch acts like a 'research lab',
TensorFlow acts like a 'production factory'.

Best systems use BOTH.
""")