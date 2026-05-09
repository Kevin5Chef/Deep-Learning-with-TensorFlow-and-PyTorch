import tensorflow as tf
import numpy as np
print("SY-5, Kevin Victor, Roll No.-30")
print("===== DELIVERY DRONE MODEL (UP TO COMPILATION) =====\n")

# =====================================================
# STEP 1: DATASET (REALISTIC DRONE FEATURES)
# =====================================================
# Features:
# [path_efficiency, flight_stability, obstacle_density,
#  payload_weight, battery_level, weather_severity]

np.random.seed(42)
samples = 300

path_eff = np.random.uniform(0.4, 1.0, samples)     # navigation efficiency
stability = np.random.uniform(0.5, 1.0, samples)    # IMU-based stability
obstacles = np.random.uniform(0.0, 1.0, samples)    # obstacle density
payload = np.random.uniform(0.5, 5.0, samples)      # kg (payload capacity impact)
battery = np.random.uniform(0.3, 1.0, samples)      # battery level
weather = np.random.uniform(0.0, 1.0, samples)      # wind/rain severity

X = np.column_stack([
    path_eff, stability, obstacles,
    payload, battery, weather
]).astype(np.float32)

# Labels: 1 = Successful Delivery, 0 = Failure
y = ((path_eff > 0.6) &
     (stability > 0.6) &
     (obstacles < 0.6) &
     (battery > 0.5) &
     (weather < 0.7)).astype(int)

print("# Dataset Shape:", X.shape)

# =====================================================
# STEP 2: PREPROCESSING (NORMALIZATION)
# =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_norm = (X - mean) / std

# =====================================================
# STEP 3: FEATURE ENGINEERING (CRITICAL)
# =====================================================
pe = X_norm[:,0]
st = X_norm[:,1]
ob = X_norm[:,2]
pl = X_norm[:,3]
bt = X_norm[:,4]
wt = X_norm[:,5]

# Derived intelligent features
navigation_score = pe * bt
risk_factor = ob + wt
load_efficiency = pl / (bt + 1e-5)

# Final features (compressed representation)
X_final = np.column_stack([
    navigation_score,
    risk_factor,
    load_efficiency
]).astype(np.float32)

# =====================================================
# STEP 4: MODEL BUILDING (KERAS)
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# =====================================================
# STEP 5: MODEL COMPILATION
# =====================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STEP 6: PRINT CONFIGURATION
# =====================================================
print("\n===== MODEL CONFIGURATION =====\n")

print("Optimizer:", model.optimizer.__class__.__name__)
print("Loss Function:", model.loss)
print("Metrics:", model.metrics_names)

print("\nModel Summary:\n")
model.summary()

print("\n===== EXPLANATION =====\n")

print("Why model compilation matters:\n")

print("Model compilation defines how the drone learns:\n")

print("Optimizer (Adam)")
print("-> Adjusts drone decision parameters (e.g., route correction, stability tuning)\n")

print("Loss Function (Crossentropy)")
print("-> Measures how wrong predictions are (e.g., predicting safe when collision risk exists)\n")

print("Metrics (Accuracy)")
print("-> Evaluates performance (successful vs failed deliveries)\n")

print("If we DO NOT compile:")
print("No error calculation")
print("No weight updates")
print("No learning\n")

print("The drone would:")
print("Fly 'blind' — making random decisions without improving")