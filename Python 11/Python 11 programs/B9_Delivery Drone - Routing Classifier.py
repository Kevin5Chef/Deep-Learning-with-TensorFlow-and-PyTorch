import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time

print("SY-5, Kevin Victor, Roll No.-30")
print("===== DRONE ROUTING MODEL (BINARY CLASSIFICATION) =====\n")

# =====================================================
# STEP 1: DATASET (ROUTING CONDITIONS)
# =====================================================
# Features:
# [distance_diff, obstacle_diff, wind_diff,
#  congestion_diff, time_pressure, safety_margin]

np.random.seed(42)
samples = 500

distance_diff = np.random.uniform(-1, 1, samples)     # shorter path advantage
obstacle_diff = np.random.uniform(-1, 1, samples)     # obstacle difference
wind_diff = np.random.uniform(-1, 1, samples)         # wind resistance
congestion_diff = np.random.uniform(-1, 1, samples)   # congestion level
time_pressure = np.random.uniform(0, 1, samples)      # urgency
safety_margin = np.random.uniform(0, 1, samples)      # safety buffer

X = np.column_stack([
    distance_diff,
    obstacle_diff,
    wind_diff,
    congestion_diff,
    time_pressure,
    safety_margin
]).astype(np.float32)

# Label: 1 = Route C, 0 = Route D
y = (
    (distance_diff < 0.2) &
    (obstacle_diff < 0.3) &
    (wind_diff < 0.3) &
    ((time_pressure > 0.5) | (safety_margin > 0.6))
).astype(int)

print("# Dataset Shape:", X.shape)

# =====================================================
# STEP 2: SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# STEP 3: NORMALIZATION
# =====================================================
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# =====================================================
# STEP 4: FEATURE ENGINEERING
# =====================================================
def engineer_features(data):
    d, o, w, c, t, s = data.T

    # Tradeoff features
    efficiency_score = -d * t
    risk_score = o + w + c
    safety_efficiency = s / (risk_score + 1e-5)

    return np.column_stack([
        efficiency_score,
        risk_score,
        safety_efficiency
    ])

X_train_f = engineer_features(X_train_n)
X_test_f = engineer_features(X_test_n)

# =====================================================
# STEP 5: MODEL (SIGMOID BINARY)
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# =====================================================
# STEP 6: COMPILATION
# =====================================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STEP 7: TRAINING
# =====================================================
print("\n===== TRAINING =====\n")

model.fit(
    X_train_f, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test_f, y_test),
    verbose=1
)

# =====================================================
# STEP 8: EVALUATION
# =====================================================
print("\n===== EVALUATION =====\n")

loss, acc = model.evaluate(X_test_f, y_test, verbose=0)

print("Test Loss:", loss)
print("Test Accuracy:", acc)

# =====================================================
# STEP 9: REAL-TIME SIMULATION (7 DECISION POINTS)
# =====================================================
print("\n===== REAL-TIME ROUTING SIMULATION =====\n")

def generate_decision(dp_input):
    dp_n = (dp_input - mean) / std
    dp_f = engineer_features(dp_n.reshape(1, -1))

    prob = model.predict(dp_f, verbose=0)[0][0]
    route = "C" if prob > 0.5 else "D"

    return route, prob

def generate_report(dp_input, route):
    d, o, w, c, t, s = dp_input

    reasons = []

    if d < 0:
        reasons.append("Shorter path selected")
    if o < 0.3:
        reasons.append("Low obstacle density")
    if w < 0.3:
        reasons.append("Favorable wind conditions")
    if t > 0.6:
        reasons.append("High urgency (fast delivery)")
    if s > 0.6:
        reasons.append("Strong safety margin")

    if not reasons:
        reasons.append("Balanced tradeoff decision")

    insight = (
        "Optimized for speed and efficiency"
        if route == "C"
        else "Optimized for safety and stability"
    )

    return reasons, insight

# Simulate 7 decision points
for dp in range(1, 8):
    print(f"\n--- Decision Point {dp} ---")

    # Simulated real-time environment
    dp_input = np.array([
        np.random.uniform(-1, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0, 1)
    ], dtype=np.float32)

    route, prob = generate_decision(dp_input)
    reasons, insight = generate_report(dp_input, route)

    print("Input:", dp_input)
    print("Chosen Route:", route)
    print("Reasons:", ", ".join(reasons))
    print("Insight:", insight)

    time.sleep(1)  # simulate 1-second real-time step