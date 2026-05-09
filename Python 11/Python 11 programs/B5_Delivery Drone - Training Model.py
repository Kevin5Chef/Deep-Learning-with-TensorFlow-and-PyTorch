import tensorflow as tf
import numpy as np

print("SY-5, Kevin Victor, Roll No.-30")
print("===== DELIVERY DRONE MODEL (WITH TRAINING) =====\n")

# =====================================================
# STEP 1: DATASET
# =====================================================
np.random.seed(42)
samples = 300

path_eff = np.random.uniform(0.4, 1.0, samples)
stability = np.random.uniform(0.5, 1.0, samples)
obstacles = np.random.uniform(0.0, 1.0, samples)
payload = np.random.uniform(0.5, 5.0, samples)
battery = np.random.uniform(0.3, 1.0, samples)
weather = np.random.uniform(0.0, 1.0, samples)

X = np.column_stack([
    path_eff, stability, obstacles,
    payload, battery, weather
]).astype(np.float32)

y = ((path_eff > 0.6) &
     (stability > 0.6) &
     (obstacles < 0.6) &
     (battery > 0.5) &
     (weather < 0.7)).astype(int)

print("# Dataset Shape:", X.shape)

# =====================================================
# STEP 2: NORMALIZATION
# =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_norm = (X - mean) / std

# =====================================================
# STEP 3: FEATURE ENGINEERING
# =====================================================
pe = X_norm[:,0]
st = X_norm[:,1]
ob = X_norm[:,2]
pl = X_norm[:,3]
bt = X_norm[:,4]
wt = X_norm[:,5]

navigation_score = pe * bt
risk_factor = ob + wt
load_efficiency = pl / (bt + 1e-5)

X_final = np.column_stack([
    navigation_score,
    risk_factor,
    load_efficiency
]).astype(np.float32)

# =====================================================
# STEP 4: MODEL
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# =====================================================
# STEP 5: COMPILATION
# =====================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STEP 6: CONFIG
# =====================================================
print("\n===== MODEL CONFIGURATION =====\n")

print("Optimizer:", model.optimizer.__class__.__name__)
print("Loss Function:", model.loss)
print("Metrics:", model.metrics_names)

print("\nModel Summary:\n")
model.summary()

# =====================================================
# STEP 7: TRAINING (AUTODIFF + BACKPROP)
# =====================================================
print("\n===== TRAINING MODEL =====\n")

history = model.fit(
    X_final, y,
    epochs=20,
    batch_size=16,
    verbose=1   # Shows progress
)

# =====================================================
# STEP 8: TRAINING EXPLANATION
# =====================================================
print("\n===== TRAINING EXPLANATION =====\n")

print("During training, the model learns from data using the following steps:\n")

print("1. Forward Pass:")
print("The drone model takes input features (navigation, risk, load) and predicts delivery success.\n")

print("2. Loss Calculation:")
print("The loss function measures how far predictions are from actual outcomes.\n")

print("3. Backpropagation:")
print("TensorFlow automatically computes gradients using autodiff (GradientTape internally).\n")

print("4. Weight Update:")
print("The optimizer (Adam) updates weights to reduce prediction error.\n")

print("5. Epochs:")
print("Each epoch means the entire dataset is processed once. Multiple epochs improve learning.\n")

print("Result:")
print("The drone gradually improves its ability to predict safe and successful deliveries.")