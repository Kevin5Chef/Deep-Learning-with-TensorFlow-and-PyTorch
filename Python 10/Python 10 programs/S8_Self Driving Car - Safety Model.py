import tensorflow as tf
import numpy as np
import time

print("SY-5, Kevin Victor, Roll No.-30")
print("===== SELF-DRIVING CAR SAFETY MODEL (IMPROVED BALANCED) =====\n")

np.random.seed(42)

# -------------------------------
# STEP 1: BASE DATA (200 samples)
# -------------------------------
samples = 200

speed = np.random.uniform(20, 60, samples)
distance = np.random.uniform(5, 60, samples)
lane_dev = np.random.uniform(0.0, 0.5, samples)
friction = np.random.uniform(0.6, 1.0, samples)
visibility = np.random.uniform(0.7, 1.0, samples)
traffic = np.random.uniform(0.1, 0.7, samples)

X_base = np.column_stack([speed, distance, lane_dev, friction, visibility, traffic])

# -------------------------------
# STEP 2: ADD 150 CLEARLY SAFE SAMPLES
# -------------------------------
safe_samples = 150

safe_speed = np.random.uniform(20, 40, safe_samples)
safe_distance = np.random.uniform(25, 60, safe_samples)
safe_lane = np.random.uniform(0.0, 0.2, safe_samples)
safe_friction = np.random.uniform(0.8, 1.0, safe_samples)
safe_visibility = np.random.uniform(0.85, 1.0, safe_samples)
safe_traffic = np.random.uniform(0.1, 0.4, safe_samples)

X_safe = np.column_stack([
    safe_speed, safe_distance, safe_lane,
    safe_friction, safe_visibility, safe_traffic
])

# Combine datasets
X = np.vstack([X_base, X_safe]).astype(np.float32)

print("# Total Dataset shape:", X.shape)

# -------------------------------
# STEP 3: LABELS (BALANCED LOGIC)
# -------------------------------
y = ((X[:,1] > 12) &          # distance
     (X[:,0] < 55) &          # speed
     (X[:,2] < 0.35) &        # lane deviation
     (X[:,3] > 0.65)).astype(np.float32)

# -------------------------------
# STEP 4: NORMALIZATION
# -------------------------------
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_norm = (X - mean) / std

# -------------------------------
# STEP 5: FEATURE ENGINEERING
# -------------------------------
s = X_norm[:,0]
d = X_norm[:,1]
l = X_norm[:,2]
f = X_norm[:,3]
v = X_norm[:,4]
t = X_norm[:,5]

risk = s / (d + 1e-5)
stability = f * v
lane_risk = l * t

X_eng = np.column_stack([X_norm, risk, stability, lane_risk])

# -------------------------------
# STEP 6: MODEL
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(9,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(0.01)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# -------------------------------
# STEP 7: TRAINING
# -------------------------------
print("\n===== TRAINING =====\n")

X_tf = tf.constant(X_eng)
y_tf = tf.constant(y.reshape(-1,1))

for epoch in range(60):
    with tf.GradientTape() as tape:
        pred = model(X_tf)
        loss = loss_fn(y_tf, pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 15 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

# -------------------------------
# ACTION FUNCTION
# -------------------------------
def decide_action(speed, distance, lane_dev, friction, visibility, traffic):
    if distance < 10:
        return "BRAKE HARD"
    elif speed > 50:
        return "SLOW DOWN"
    elif lane_dev > 0.35:
        return "STEER CORRECT"
    elif traffic > 0.6:
        return "REDUCE SPEED"
    elif visibility < 0.75:
        return "CAUTION MODE"
    else:
        return "MAINTAIN SPEED"

# -------------------------------
# EXPLANATION FUNCTION
# -------------------------------
def explain_conditions(speed, distance, lane_dev, friction, visibility, traffic):
    reasons = []

    if distance < 12:
        reasons.append("Close obstacle")
    elif speed > 50:
        reasons.append("Higher speed")
    elif lane_dev > 0.35:
        reasons.append("Lane drift")
    elif friction < 0.7:
        reasons.append("Low grip")
    elif visibility < 0.75:
        reasons.append("Reduced visibility")
    elif traffic > 0.6:
        reasons.append("Heavy traffic")
    else:
        reasons.append("Stable driving conditions")

    return reasons

# -------------------------------
# STEP 8: REAL-TIME SIMULATION
# -------------------------------
print("\n===== REAL-TIME SIMULATION (BALANCED) =====\n")

for t in range(1, 11):

    # Mix safe & unsafe scenarios intentionally
    if t % 2 == 0:
        # SAFE scenario
        test = np.array([[
            np.random.uniform(25, 45),
            np.random.uniform(20, 50),
            np.random.uniform(0.05, 0.25),
            np.random.uniform(0.8, 1.0),
            np.random.uniform(0.85, 1.0),
            np.random.uniform(0.2, 0.5)
        ]], dtype=np.float32)
    else:
        # UNSAFE scenario
        test = np.array([[
            np.random.uniform(45, 60),
            np.random.uniform(5, 15),
            np.random.uniform(0.3, 0.5),
            np.random.uniform(0.6, 0.75),
            np.random.uniform(0.7, 0.85),
            np.random.uniform(0.5, 0.7)
        ]], dtype=np.float32)

    test_n = (test - mean) / std

    s, d, l, f, v, tr = test_n[0]

    rf = s / (d + 1e-5)
    st = f * v
    lr = l * tr

    test_eng = np.column_stack([test_n, [rf], [st], [lr]])

    pred = model(tf.constant(test_eng))
    score = pred.numpy()[0][0]

    decision = "SAFE" if score > 0.5 else "UNSAFE"

    action = decide_action(*test[0])
    reasons = explain_conditions(*test[0])

    print(f"\nTime: {t} sec")
    print("Sensor Data:", test[0])
    print(f"Prediction Score: {score:.3f}")
    print("Decision:", decision)
    print("Action:", action)
    print("Reason:", ", ".join(reasons))

    time.sleep(1)