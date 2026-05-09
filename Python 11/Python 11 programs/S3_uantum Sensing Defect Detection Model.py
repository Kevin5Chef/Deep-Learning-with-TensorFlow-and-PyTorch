import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

print("SY-5, Kevin Victor, Roll No.-30")
print("===== QUANTUM SENSING DEFECT DETECTION MODEL =====\n")

# =====================================================
# STEP 1: DATASET (BALANCED, NOISE-AWARE)
# =====================================================
# Features:
# [thermal_variation, electron_noise, EM_interference,
#  lattice_vibration, signal_strength, measurement_uncertainty]

np.random.seed(42)
samples = 600

# Defect-free samples
clean = 300
thermal_c = np.random.uniform(0.0, 0.3, clean)
electron_c = np.random.uniform(0.0, 0.3, clean)
em_c = np.random.uniform(0.0, 0.3, clean)
lattice_c = np.random.uniform(0.0, 0.3, clean)
signal_c = np.random.uniform(0.7, 1.0, clean)
uncertainty_c = np.random.uniform(0.0, 0.2, clean)

# Defective samples
defect = 300
thermal_d = np.random.uniform(0.5, 1.0, defect)
electron_d = np.random.uniform(0.5, 1.0, defect)
em_d = np.random.uniform(0.5, 1.0, defect)
lattice_d = np.random.uniform(0.5, 1.0, defect)
signal_d = np.random.uniform(0.2, 0.6, defect)
uncertainty_d = np.random.uniform(0.4, 1.0, defect)

# Combine
X = np.vstack([
    np.column_stack([thermal_c, electron_c, em_c, lattice_c, signal_c, uncertainty_c]),
    np.column_stack([thermal_d, electron_d, em_d, lattice_d, signal_d, uncertainty_d])
]).astype(np.float32)

# Labels: 0 = CLEAN, 1 = DEFECT
y = np.array([0]*clean + [1]*defect)

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
# STEP 4: FEATURE ENGINEERING (NOISE FILTERING)
# =====================================================
def engineer_features(data):
    t, e, em, lv, s, u = data.T

    # Signal processing inspired features
    signal_to_noise = s / (t + e + em + lv + u + 1e-5)
    noise_profile = t + e + em + lv
    stability_index = s * (1 - u)

    return np.column_stack([
        signal_to_noise,
        noise_profile,
        stability_index
    ])

X_train_f = engineer_features(X_train_n)
X_test_f = engineer_features(X_test_n)

# =====================================================
# STEP 5: MODEL (MULTIPLE HIDDEN LAYERS)
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),   # SECOND HIDDEN LAYER
    tf.keras.layers.Dense(8, activation='relu'),    # THIRD HIDDEN LAYER
    tf.keras.layers.Dense(1, activation='sigmoid')
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

def generate_provenance(sample, confidence):
    t, e, em, lv, s, u = sample

    reasons = []

    # Reasoning based on thresholds
    if t > 0.5:
        reasons.append("High thermal variation")
    if e > 0.5:
        reasons.append("Electron noise detected")
    if em > 0.5:
        reasons.append("Electromagnetic interference")
    if lv > 0.5:
        reasons.append("Lattice instability")
    if s < 0.6:
        reasons.append("Weak signal strength")
    if u > 0.4:
        reasons.append("High measurement uncertainty")

    if not reasons:
        reasons.append("Stable quantum conditions")

    # Insight generation
    if confidence > 0.8:
        insight = "High confidence prediction with strong signal clarity"
    elif confidence > 0.6:
        insight = "Moderate confidence, some noise influence present"
    else:
        insight = "Low confidence due to conflicting signal patterns"

    return reasons, insight

# =====================================================
# STEP 9: SAMPLE PREDICTIONS
# =====================================================
print("\n===== SAMPLE ANALYSIS =====\n")

test_samples = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.9, 0.1],  # clean
    [0.7, 0.8, 0.7, 0.6, 0.4, 0.7],  # defect
    [0.3, 0.2, 0.3, 0.2, 0.8, 0.2],  # borderline
], dtype=np.float32)

test_n = (test_samples - mean) / std
test_f = engineer_features(test_n)

preds = model.predict(test_f)

preds = model.predict(test_f)

for i in range(len(test_samples)):
    confidence = preds[i][0]
    decision = "DEFECT" if confidence > 0.5 else "CLEAN"

    reasons, insight = generate_provenance(test_samples[i], confidence)

    print(f"\nSample {i+1}")
    print("Input:", test_samples[i])
    print("Prediction:", decision)

    print("Reasons:", ", ".join(reasons))


# =====================================================
# STEP 10: EXPLANATION (LAYERS IMPORTANCE)
# =====================================================
print("\n===== EXPLANATION =====\n")

print("Why multiple hidden layers improve performance:\n")

print("1. First Hidden Layer:")
print("Learns basic signal patterns such as noise levels and signal strength.\n")

print("2. Second Hidden Layer:")
print("Combines basic patterns to detect complex interactions like electron interference and thermal instability.\n")

print("3. Third Hidden Layer:")
print("Extracts high-level abstractions such as defect signatures vs clean signals.\n")

print("Overall Impact:")
print("Multiple layers enable deep feature extraction, better noise filtering, and accurate detection of nanometer-level defects.\n")

print("Without multiple layers:")
print("The model would fail to capture complex quantum interactions and misclassify noisy signals.")