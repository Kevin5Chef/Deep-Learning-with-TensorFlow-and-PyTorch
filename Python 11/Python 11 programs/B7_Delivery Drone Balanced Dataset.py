import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

print("SY-5, Kevin Victor, Roll No.-30")
print("===== DELIVERY DRONE MODEL (BALANCED DATASET) =====\n")

# =====================================================
# STEP 1: DATASET (BALANCED)
# =====================================================
np.random.seed(42)

# -------- ORIGINAL RANDOM DATA --------
samples = 300

path_eff = np.random.uniform(0.4, 1.0, samples)
stability = np.random.uniform(0.5, 1.0, samples)
obstacles = np.random.uniform(0.0, 1.0, samples)
payload = np.random.uniform(0.5, 5.0, samples)
battery = np.random.uniform(0.3, 1.0, samples)
weather = np.random.uniform(0.0, 1.0, samples)

X_random = np.column_stack([
    path_eff, stability, obstacles,
    payload, battery, weather
]).astype(np.float32)

# -------- ADD 200 SUCCESS-ORIENTED SAMPLES --------
success_samples = 200

path_eff_s = np.random.uniform(0.75, 1.0, success_samples)
stability_s = np.random.uniform(0.75, 1.0, success_samples)
obstacles_s = np.random.uniform(0.0, 0.3, success_samples)
payload_s = np.random.uniform(0.5, 3.0, success_samples)
battery_s = np.random.uniform(0.7, 1.0, success_samples)
weather_s = np.random.uniform(0.0, 0.4, success_samples)

X_success = np.column_stack([
    path_eff_s, stability_s, obstacles_s,
    payload_s, battery_s, weather_s
]).astype(np.float32)

# -------- COMBINE --------
X = np.vstack([X_random, X_success])

# Labels
y = ((X[:,0] > 0.6) &
     (X[:,1] > 0.6) &
     (X[:,2] < 0.6) &
     (X[:,4] > 0.5) &
     (X[:,5] < 0.7)).astype(int)

print("# Dataset Shape:", X.shape)
print("# Success Samples:", np.sum(y == 1))
print("# Failure Samples:", np.sum(y == 0))

# =====================================================
# STEP 2: TRAIN-TEST SPLIT
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
# STEP 4: FEATURE ENGINEERING (IMPROVED)
# =====================================================
def engineer_features(data):
    pe, st, ob, pl, bt, wt = data.T

    # Core features
    navigation_score = pe * bt
    risk_factor = ob + wt
    load_efficiency = pl / (bt + 1e-5)

    # NEW SUCCESS-ORIENTED FEATURES
    stability_margin = st * bt
    safety_index = (1 - ob) * (1 - wt)
    efficiency_balance = pe * st / (pl + 1e-5)

    return np.column_stack([
        navigation_score,
        risk_factor,
        load_efficiency,
        stability_margin,
        safety_index,
        efficiency_balance
    ])

X_train_f = engineer_features(X_train_n)
X_test_f = engineer_features(X_test_n)

# =====================================================
# STEP 5: MODEL
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(6,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# =====================================================
# STEP 6: COMPILATION
# =====================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STEP 7: TRAINING
# =====================================================
print("\n===== TRAINING =====\n")

model.fit(
    X_train_f, y_train,
    epochs=25,
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
# STEP 9: PROVENANCE REPORT
# =====================================================
def generate_report(sample, prediction):
    pe, st, ob, pl, bt, wt = sample

    reasons = []

    if pe < 0.6:
        reasons.append("Inefficient navigation")
    if st < 0.6:
        reasons.append("Low stability")
    if ob > 0.6:
        reasons.append("Obstacle congestion")
    if bt < 0.5:
        reasons.append("Battery constraint")
    if wt > 0.7:
        reasons.append("Severe weather")

    if not reasons:
        reasons.append("Strong operational conditions")

    decision = "TRIP SUCCESS" if prediction == 1 else "TRIP FAILURE"

    insight = (
        "System operating optimally with high confidence"
        if prediction == 1
        else "Operational risks dominate decision space"
    )

    return decision, reasons, insight

# =====================================================
# STEP 10: TEST SAMPLES
# =====================================================
print("\n===== SAMPLE PREDICTIONS =====\n")

test_samples = np.array([
    [0.8, 0.9, 0.2, 2.0, 0.9, 0.3],
    [0.5, 0.6, 0.7, 3.0, 0.6, 0.8],
    [0.7, 0.8, 0.4, 4.5, 0.7, 0.5],
    [0.4, 0.5, 0.8, 3.5, 0.4, 0.9],
    [0.9, 0.95, 0.1, 1.5, 0.95, 0.2]
], dtype=np.float32)

# Preprocess
test_n = (test_samples - mean) / std
test_f = engineer_features(test_n)

pred_probs = model.predict(test_f)
pred_classes = np.argmax(pred_probs, axis=1)

# Display
for i in range(5):
    decision, reasons, insight = generate_report(test_samples[i], pred_classes[i])

    print(f"\n--- Sample {i+1} ---")
    print("Input:", test_samples[i])
    print("Prediction:", decision)
    print("Reasons:", ", ".join(reasons))
    print("Insight:", insight)