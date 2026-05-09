import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
print("SY-5, Kevin Victor, Roll No.-30")
print("===== ACS DRONE INSPECTION MODEL =====\n")

# =====================================================
# STEP 1: DATASET (REALISTIC + DEC LOGIC)
# =====================================================
np.random.seed(42)
samples = 500

# Features:
# [task_completion, weather_clear, wind_stability,
#  battery_level, time_window, obstacle_risk]

task = np.random.uniform(0.5, 1.0, samples)
weather = np.random.uniform(0.3, 1.0, samples)
wind = np.random.uniform(0.2, 1.0, samples)
battery = np.random.uniform(0.3, 1.0, samples)
time_window = np.random.uniform(0.3, 1.0, samples)
obstacles = np.random.uniform(0.0, 1.0, samples)

X = np.column_stack([
    task, weather, wind,
    battery, time_window, obstacles
]).astype(np.float32)

# =====================================================
# STEP 2: DYNAMIC EDGE CASE LABELING
# =====================================================
y = []

for i in range(samples):
    t, w, wi, b, tw, o = X[i]

    # Strict unsafe conditions
    if (t < 0.6 and tw < 0.5) or (wi < 0.3 and w < 0.5) or (o > 0.8):
        y.append(0)

    # Dynamic Edge Case (DEC)
    elif (wi < 0.5 and b > 0.7) or (tw < 0.5 and t > 0.7):
        y.append(1)

    # Normal feasible
    elif (t > 0.7 and w > 0.6 and wi > 0.5 and b > 0.5):
        y.append(1)

    else:
        y.append(1)

y = np.array(y)

# =====================================================
# STEP 3: SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# STEP 4: NORMALIZATION
# =====================================================
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# =====================================================
# STEP 5: FEATURE ENGINEERING
# =====================================================
def engineer(data):
    t, w, wi, b, tw, o = data.T

    readiness = t * w
    stability = wi * b
    time_efficiency = tw * t
    risk = o * (1 - wi)

    return np.column_stack([
        readiness,
        stability,
        time_efficiency,
        risk
    ])

X_train_f = engineer(X_train_n)
X_test_f = engineer(X_test_n)

# =====================================================
# STEP 6: MODEL (SIGMOID)
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# =====================================================
# STEP 7: COMPILATION
# =====================================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STEP 8: TRAINING (AUTODIFF + BACKPROP)
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
# STEP 9: EVALUATION
# =====================================================
print("\n===== EVALUATION =====\n")

loss, acc = model.evaluate(X_test_f, y_test, verbose=0)

print("Test Accuracy:", acc)
print("Test Loss:", loss)

# =====================================================
# STEP 10: TEST SCENARIOS (UPDATED WITH DANGEROUS CASES)
# =====================================================
test_cases = np.array([
    [0.9, 0.8, 0.7, 0.8, 0.9, 0.2],  # ideal
    [0.6, 0.7, 0.4, 0.9, 0.6, 0.3],  # DEC (battery compensates)
    [0.8, 0.5, 0.3, 0.8, 0.7, 0.2],  # DEC wind
    [0.5, 0.4, 0.2, 0.6, 0.4, 0.5],  # borderline
    [0.7, 0.8, 0.6, 0.5, 0.4, 0.2],  # DEC time window
    [0.9, 0.9, 0.8, 0.9, 0.9, 0.1],  # perfect
    [0.6, 0.3, 0.2, 0.4, 0.3, 0.9],  # high risk
    [0.4, 0.2, 0.1, 0.3, 0.2, 0.95], # EXTREME: low task, bad weather, unstable wind, high obstacles
    [0.5, 0.3, 0.2, 0.2, 0.3, 0.85]  # EXTREME: low battery + poor weather + high risk
], dtype=np.float32)

# =====================================================
# STEP 11: REAL-TIME SIMULATION
# =====================================================
print("\n===== REAL-TIME INSPECTION SIMULATION =====\n")

for i in range(len(test_cases)):
    test = test_cases[i].reshape(1, -1)

    test_n = (test - mean) / std
    test_f = engineer(test_n)

    pred = model.predict(test_f, verbose=0)[0][0]

    decision = "FEASIBLE" if pred > 0.5 else "NOT FEASIBLE"

    t, w, wi, b, tw, o = test[0]

    reasons = []
    plan = []

    # Reasoning
    if decision == "FEASIBLE":
        if wi < 0.5 and b > 0.7:
            reasons.append("Wind compensated by high battery (DEC)")
        elif tw < 0.5:
            reasons.append("Limited time, but prioritizing coverage (DEC)")
        else:
            reasons.append("All operational conditions stable")

        plan = [
            "Initiate drone takeoff",
            "Perform inspection sweep",
            "Capture essential structural data"
        ]

    else:
        if o > 0.8:
            reasons.append("Obstacle risk too high")
        if wi < 0.3:
            reasons.append("Wind too unstable")
        if w < 0.5:
            reasons.append("Weather conditions unsafe")

        plan = [
            "Abort mission",
            "Reschedule inspection",
            "Re-evaluate conditions"
        ]

    print(f"\nTime {i+1}s")
    print("Input:", test[0])
    print("Decision:", decision)
    print("Reasons:", ", ".join(reasons))
    print("Plan:")
    for step in plan:
        print("-", step)

    time.sleep(1)