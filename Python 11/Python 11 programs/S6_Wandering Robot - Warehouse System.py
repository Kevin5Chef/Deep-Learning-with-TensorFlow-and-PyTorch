import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time

print("SY-5, Kevin Victor, Roll No.-30")
print("===== WANDERING ROBOT (WR) SYSTEM =====\n")

# =====================================================
# STEP 1: DATASET
# =====================================================
np.random.seed(42)
samples = 500

weight = np.random.uniform(0.1, 10, samples)
fragility = np.random.uniform(0, 1, samples)
reflectivity = np.random.uniform(0, 1, samples)
hazard = np.random.uniform(0, 1, samples)
clutter = np.random.uniform(0, 1, samples)
lighting = np.random.uniform(0, 1, samples)

X = np.column_stack([
    weight, fragility, reflectivity,
    hazard, clutter, lighting
]).astype(np.float32)

y = (
    (fragility < 0.6) &
    (hazard < 0.6) &
    (clutter < 0.7) &
    (weight < 7)
).astype(int)

# =====================================================
# SPLIT + NORMALIZE
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def engineer(data):
    w, f, r, h, c, l = data.T
    return np.column_stack([
        w * f,
        l * r,
        h + c
    ])

X_train_f = engineer(X_train_n)
X_test_f = engineer(X_test_n)

# =====================================================
# MODEL
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# TRAINING
# =====================================================
model.fit(X_train_f, y_train, epochs=15, batch_size=16, verbose=0)

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time

print("SY-5, Kevin Victor, Roll No.-30")
print("===== WANDERING ROBOT (WR) SYSTEM =====\n")

# =====================================================
# STEP 1: DATASET
# =====================================================
np.random.seed(42)
samples = 500

weight = np.random.uniform(0.1, 10, samples)
fragility = np.random.uniform(0, 1, samples)
reflectivity = np.random.uniform(0, 1, samples)
hazard = np.random.uniform(0, 1, samples)
clutter = np.random.uniform(0, 1, samples)
lighting = np.random.uniform(0, 1, samples)

X = np.column_stack([
    weight, fragility, reflectivity,
    hazard, clutter, lighting
]).astype(np.float32)

y = (
    (fragility < 0.6) &
    (hazard < 0.6) &
    (clutter < 0.7) &
    (weight < 7)
).astype(int)

# =====================================================
# SPLIT + NORMALIZE
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def engineer(data):
    w, f, r, h, c, l = data.T
    return np.column_stack([
        w * f,
        l * r,
        h + c
    ])

X_train_f = engineer(X_train_n)
X_test_f = engineer(X_test_n)

# =====================================================
# MODEL
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# TRAINING
# =====================================================
model.fit(X_train_f, y_train, epochs=15, batch_size=16, verbose=0)

# =====================================================
# PROVENANCE
# =====================================================
def report(sample):
    w, f, r, h, c, l = sample

    reasons = []
    if f > 0.6: reasons.append("High fragility")
    if h > 0.6: reasons.append("Electrical hazard")
    if c > 0.7: reasons.append("High clutter")
    if l < 0.3: reasons.append("Low visibility")

    if not reasons:
        reasons.append("Safe handling conditions")

    return reasons

# =====================================================
# SIMULATION
# =====================================================
print("\n===== REAL-TIME SIMULATION =====\n")

memory = []           # stores explored objects
plan = []             # stores planned actions

for t in range(30):

    # -------------------------------
    # PHASE CONTROL
    # -------------------------------
    if t < 10:
        phase = "Exploration"
    elif t < 20:
        phase = "Learning"
    elif t < 25:
        phase = "Planning"
    else:
        phase = "Execution"

    print(f"\nTime {t+1}s | Phase: {phase}")

    # =====================================================
    # EXPLORATION + LEARNING (RANDOM ENVIRONMENT)
    # =====================================================
    if phase in ["Exploration", "Learning"]:

        sample = np.array([
            np.random.uniform(0.1, 10),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1)
        ], dtype=np.float32)

        sample_n = (sample - mean) / std
        sample_f = engineer(sample_n.reshape(1, -1))

        pred = model.predict(sample_f, verbose=0)[0][0]

        if phase == "Exploration":
            pred = np.random.uniform(0, 1)

        decision = "PICK" if pred > 0.5 else "SKIP"

        memory.append((sample, decision))

        print("Object:", sample)
        print("Decision:", decision)
        print("Confidence:", round(pred, 3))
        print("Reasons:", ", ".join(report(sample)))

    # =====================================================
    # PLANNING PHASE (CREATE TASK LIST)
    # =====================================================
    elif phase == "Planning":

        if not plan:
            print("Generating plan from learned experience...")

            for obj, decision in memory:
                if decision == "PICK":
                    plan.append(("PICK", obj))
                    plan.append(("PLACE", obj))

            print("Plan Created:", len(plan), "actions")

        print("Planning: Organizing objects by safety and type")

    # =====================================================
    # EXECUTION PHASE (DETERMINISTIC)
    # =====================================================
    else:

        if plan:
            action, obj = plan.pop(0)

            print("Planned Object:", obj)
            print("Action:", action)

            reasons = report(obj)

            print("Reasons:", ", ".join(reasons))

            if action == "PICK":
                print("Execution: Safely picking object")
            else:
                print("Execution: Placing object in designated slot")

        else:
            print("Execution Complete: Warehouse organized successfully")

    time.sleep(1)

# =====================================================
# PROVENANCE
# =====================================================
def report(sample):
    w, f, r, h, c, l = sample

    reasons = []
    if f > 0.6: reasons.append("High fragility")
    if h > 0.6: reasons.append("Electrical hazard")
    if c > 0.7: reasons.append("High clutter")
    if l < 0.3: reasons.append("Low visibility")

    if not reasons:
        reasons.append("Safe handling conditions")

    return reasons

# =====================================================
# SIMULATION
# =====================================================
print("\n===== REAL-TIME SIMULATION =====\n")

memory = []           # stores explored objects
plan = []             # stores planned actions

for t in range(30):

    # -------------------------------
    # PHASE CONTROL
    # -------------------------------
    if t < 10:
        phase = "Exploration"
    elif t < 20:
        phase = "Learning"
    elif t < 25:
        phase = "Planning"
    else:
        phase = "Execution"

    print(f"\nTime {t+1}s | Phase: {phase}")

    # =====================================================
    # EXPLORATION + LEARNING (RANDOM ENVIRONMENT)
    # =====================================================
    if phase in ["Exploration", "Learning"]:

        sample = np.array([
            np.random.uniform(0.1, 10),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1)
        ], dtype=np.float32)

        sample_n = (sample - mean) / std
        sample_f = engineer(sample_n.reshape(1, -1))

        pred = model.predict(sample_f, verbose=0)[0][0]

        if phase == "Exploration":
            pred = np.random.uniform(0, 1)

        decision = "PICK" if pred > 0.5 else "SKIP"

        memory.append((sample, decision))

        print("Object:", sample)
        print("Decision:", decision)
        print("Confidence:", round(pred, 3))
        print("Reasons:", ", ".join(report(sample)))

    # =====================================================
    # PLANNING PHASE (CREATE TASK LIST)
    # =====================================================
    elif phase == "Planning":

        if not plan:
            print("Generating plan from learned experience...")

            for obj, decision in memory:
                if decision == "PICK":
                    plan.append(("PICK", obj))
                    plan.append(("PLACE", obj))

            print("Plan Created:", len(plan), "actions")

        print("Planning: Organizing objects by safety and type")

    # =====================================================
    # EXECUTION PHASE (DETERMINISTIC)
    # =====================================================
    else:

        if plan:
            action, obj = plan.pop(0)

            print("Planned Object:", obj)
            print("Action:", action)

            reasons = report(obj)



            if action == "PICK":
                print("Execution: Safely picking object")
            else:
                print("Execution: Placing object in designated slot")

        else:
            print("Execution Complete: Warehouse organized successfully")

    time.sleep(1)