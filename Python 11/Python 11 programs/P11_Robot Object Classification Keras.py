import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
print("SY-5, Kevin Victor, Roll No.-30")
print("===== HUMANOID ROBOT: LIVING ROOM LEARNING SYSTEM =====\n")

# =====================================================
# STEP 1: DATASET (SIMULATED REALISTIC ENVIRONMENT)
# =====================================================
# Features:
# [size, weight, brightness, motion, texture, reflectivity, position_stability]

# Labels (object categories):
# 0 = Furniture
# 1 = Electronics
# 2 = Decor
# 3 = Dynamic Object

X = np.array([
    # Furniture
    [8, 20, 5, 0, 7, 2, 9],   # Sofa
    [7, 15, 4, 0, 6, 2, 9],   # Table
    [6, 12, 4, 0, 6, 3, 8],   # Chair
    [9, 25, 3, 0, 8, 1, 9],   # Cabinet

    # Electronics
    [5, 8, 9, 0, 4, 8, 7],    # TV
    [3, 2, 7, 0, 3, 6, 6],    # Remote
    [4, 3, 8, 0, 4, 7, 6],    # Speaker
    [2, 1, 10, 0, 2, 9, 5],   # LED Light

    # Decor
    [2, 1, 6, 0, 5, 4, 8],    # Painting
    [3, 2, 5, 0, 6, 5, 8],    # Vase
    [4, 3, 4, 0, 7, 3, 9],    # Carpet
    [2, 1, 6, 0, 5, 4, 8],    # Ornament

    # Dynamic objects
    [5, 2, 6, 8, 5, 3, 4],    # Curtain moving
    [3, 1, 9, 7, 3, 6, 3],    # Flickering light
    [4, 2, 7, 9, 4, 5, 2],    # Toy car moving
    [6, 3, 6, 8, 6, 4, 3]     # Rain on window
], dtype=np.float32)

y = np.array([
    0,0,0,0,   # Furniture
    1,1,1,1,   # Electronics
    2,2,2,2,   # Decor
    3,3,3,3    # Dynamic
], dtype=np.int32)

print("# Dataset Shape:", X.shape)

# =====================================================
# STEP 2: TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("# Training samples:", len(X_train))
print("# Testing samples:", len(X_test))

# =====================================================
# STEP 3: MODEL BUILDING (KERAS)
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(7,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
])

# =====================================================
# STEP 4: COMPILATION
# =====================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STEP 5: TRAINING
# =====================================================
print("\n===== TRAINING MODEL =====\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    verbose=0,
    validation_data=(X_test, y_test)
)

print("Final Training Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])

# =====================================================
# STEP 6: EVALUATION
# =====================================================
print("\n===== MODEL EVALUATION =====\n")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# =====================================================
# STEP 7: PROVENANCE REPORT SYSTEM
# =====================================================

labels = ["Furniture", "Electronics", "Decor", "Dynamic Object"]

def generate_report(sample, prediction):
    report = []

    size, weight, brightness, motion, texture, reflectivity, stability = sample

    report.append(f"Predicted Category: {labels[prediction]}")

    # Observations
    if motion > 6:
        report.append("Object exhibits dynamic behavior (movement detected).")

    if brightness > 7:
        report.append("High light emission or reflection detected.")

    if stability > 7:
        report.append("Object is structurally stable and stationary.")

    if texture > 6:
        report.append("Surface texture is rough/complex.")

    if reflectivity > 6:
        report.append("Highly reflective surface observed.")

    # Learning insight
    report.append("Robot Insight: Objects with motion are likely dynamic entities.")
    report.append("Robot Insight: Stable, heavy objects belong to furniture class.")

    return "\n".join(report)

# =====================================================
# STEP 8: TESTING WITH NEW OBSERVATIONS
# =====================================================
print("\n===== ROBOT OBSERVATION & PROVENANCE REPORT =====\n")

test_samples = np.array([
    [6, 18, 4, 0, 7, 2, 9],   # Sofa-like
    [3, 1, 9, 8, 3, 6, 3],    # Flickering light
    [4, 2, 6, 9, 4, 5, 2],    # Moving toy
    [2, 1, 6, 0, 5, 4, 8]     # Ornament
], dtype=np.float32)

predictions = model.predict(test_samples)

for i, sample in enumerate(test_samples):
    pred_class = np.argmax(predictions[i])

    print(f"\n--- Observation {i+1} ---")
    print("Input Features:", sample)
    print(generate_report(sample, pred_class))