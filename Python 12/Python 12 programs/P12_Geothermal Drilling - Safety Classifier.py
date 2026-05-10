import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
print("SY-5, Kevin Victor, Roll No.-30")
print("===== GEOTHERMAL DRILLING DECISION SYSTEM (FIXED) =====\n")

np.random.seed(42)

# =====================================================
# STEP 1: BASE DATA (RANDOM)
# =====================================================
samples = 300

tg = np.random.uniform(20, 80, samples)      # thermal gradient
rs = np.random.uniform(0.3, 1.0, samples)    # rock stability
es = np.random.uniform(0.2, 1.0, samples)    # energy signal

X_base = np.column_stack([tg, rs, es])

# =====================================================
# STEP 2: ADD EXPLICIT SAFE SAMPLES
# =====================================================
safe_samples = 200

safe_tg = np.random.uniform(40, 65, safe_samples)
safe_rs = np.random.uniform(0.7, 1.0, safe_samples)
safe_es = np.random.uniform(0.6, 1.0, safe_samples)

X_safe = np.column_stack([safe_tg, safe_rs, safe_es])

# Combine
X = np.vstack([X_base, X_safe]).astype(np.float32)

# =====================================================
# STEP 3: LABELS (BALANCED)
# =====================================================
y = ((X[:,1] > 0.6) &      # stability
     (X[:,2] > 0.5) &      # energy
     (X[:,0] < 70)).astype(int)

print("# Dataset Shape:", X.shape)

# =====================================================
# STEP 4: NORMALIZATION
# =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_norm = (X - mean) / std

# =====================================================
# STEP 5: IMPROVED FEATURE ENGINEERING (CRITICAL)
# =====================================================
tg_n = X_norm[:,0]
rs_n = X_norm[:,1]
es_n = X_norm[:,2]

# New meaningful features
safety_score = rs_n - 0.5 * tg_n          # stability vs heat stress
energy_potential = es_n * tg_n            # geothermal viability
hazard_index = tg_n / (rs_n + 1e-5)       # risk factor

X_final = np.column_stack([
    safety_score,
    energy_potential,
    hazard_index
]).astype(np.float32)

# =====================================================
# STEP 6: TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
y_test_cat = tf.keras.utils.to_categorical(y_test, 2)

# =====================================================
# STEP 7: MODEL (STRICT ARCHITECTURE)
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# =====================================================
# STEP 8: COMPILE
# =====================================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STEP 9: TRAINING
# =====================================================
print("\n===== TRAINING =====\n")

history = model.fit(
    X_train, y_train_cat,
    epochs=60,
    validation_data=(X_test, y_test_cat),
    verbose=0
)

print("Final Training Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])

# =====================================================
# STEP 10: EVALUATION
# =====================================================
print("\n===== MODEL EVALUATION =====\n")

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)

print("Test Loss:", loss)
print("Test Accuracy:", acc)

# =====================================================
# STEP 11: PROVENANCE REPORT
# =====================================================
def generate_report(original, probs):
    tg, rs, es = original
    safe_prob = probs[1]

    report = []

    report.append(f"Safe Probability: {safe_prob:.2f}")

    # Productivity
    if es > 0.7 and tg > 50:
        report.append("High geothermal potential (productive).")
    elif es > 0.5:
        report.append("Moderate geothermal potential.")
    else:
        report.append("Low geothermal potential.")

    # Safety reasoning
    if rs < 0.5:
        report.append("Low rock stability.")
    if tg > 70:
        report.append("Excessive thermal stress.")
    if es < 0.4:
        report.append("Weak subsurface signals.")

    if safe_prob > 0.5:
        report.append("FINAL DECISION: SAFE TO DRILL")
    else:
        report.append("FINAL DECISION: UNSAFE TO DRILL")

    return "\n".join(report)

# =====================================================
# STEP 12: TESTING
# =====================================================
print("\n===== TESTING NEW LOCATIONS =====\n")

test_samples = np.array([
    [60, 0.8, 0.9],   # SHOULD BE SAFE
    [75, 0.4, 0.6],   # UNSAFE
    [45, 0.7, 0.5],   # BORDERLINE SAFE
    [30, 0.9, 0.3]    # SAFE BUT LOW PRODUCTIVITY
], dtype=np.float32)

# Preprocess
test_norm = (test_samples - mean) / std

tg_n, rs_n, es_n = test_norm[:,0], test_norm[:,1], test_norm[:,2]

safety_score = rs_n - 0.5 * tg_n
energy_potential = es_n * tg_n
hazard_index = tg_n / (rs_n + 1e-5)

test_final = np.column_stack([
    safety_score,
    energy_potential,
    hazard_index
])

preds = model.predict(test_final)

for i in range(len(test_samples)):
    print(f"\n--- Location {i+1} ---")
    print("Input:", test_samples[i])
    print(generate_report(test_samples[i], preds[i]))