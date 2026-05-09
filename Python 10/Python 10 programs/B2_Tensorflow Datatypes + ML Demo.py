# ============================================================
# TENSOR DATA TYPES DEMONSTRATION + ML EXAMPLE
# ============================================================
print("SY-5, Kevin Victor, Roll No.-30")
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# STEP 1: CREATE TENSORS
# -----------------------------

int_tensor = tf.constant([1, 2, 3, 4], dtype=tf.int32)
float_tensor = tf.constant([1.5, 2.7, 3.2, 4.8], dtype=tf.float32)
string_tensor = tf.constant(["battery", "fusion", "material"])

print("\n===== TENSOR VALUES =====")
print("Integer Tensor:", int_tensor)
print("Float Tensor:", float_tensor)
print("String Tensor:", string_tensor)

print("\n===== TENSOR DATA TYPES =====")
print("Integer Tensor Type:", int_tensor.dtype)
print("Float Tensor Type:", float_tensor.dtype)
print("String Tensor Type:", string_tensor.dtype)

# -----------------------------
# STEP 2: COMPARISON
# -----------------------------

print("\n===== COMPARISON =====")
print("Int values:", int_tensor.numpy())
print("Float values:", float_tensor.numpy())
print("String values:", string_tensor.numpy())

# -----------------------------
# STEP 3: WHEN TO USE WHICH TYPE
# -----------------------------

print("\n===== WHEN TO USE DATA TYPES =====")

print("""
INT32:
- Used for counts, indices, categories
- Example: number of items, class labels

FLOAT32:
- Used for continuous values
- Example: temperature, probability, voltage

STRING:
- Used for labels, names, metadata
""")

# -----------------------------
# STEP 4: SMALL ML EXAMPLE
# -----------------------------
# Predict if battery material is "safe" or "unsafe"

# Features:
# temperature (float), voltage (float), cycles (int)

temperature = np.array([300.5, 500.2, 450.1, 600.3], dtype=np.float32)
voltage = np.array([3.7, 4.1, 3.9, 4.3], dtype=np.float32)
cycles = np.array([1000, 500, 800, 300], dtype=np.int32)

# Combine features
X = np.column_stack((temperature, voltage, cycles))

# Labels (0 = unsafe, 1 = safe)
y = np.array([1, 0, 1, 0], dtype=np.int32)

print("\n===== ML DATA TYPES =====")
print("Feature matrix dtype:", X.dtype)
print("Label dtype:", y.dtype)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

print("\n===== ML MODEL OUTPUT =====")
print("Predictions:", predictions)
print("Accuracy:", accuracy)

# -----------------------------
# STEP 5: EXPLANATION
# -----------------------------

print("\n===== EXPLANATION =====")

print("""
1. Why FLOAT32 for features?
   - Temperature and voltage are continuous values
   - Model needs precision (decimals matter)

2. Why INT32 for labels?
   - Labels are discrete categories (safe/unsafe)
   - No need for decimals

3. Why not INT32 for features?
   - You lose precision (300.5 → 300)
   - Model becomes less accurate

4. Why not FLOAT32 for labels?
   - Labels are categorical, not continuous
   - Float labels can confuse classification models

5. Key Insight:
   - Features → FLOAT32
   - Labels → INT32
   - Metadata → STRING
""")