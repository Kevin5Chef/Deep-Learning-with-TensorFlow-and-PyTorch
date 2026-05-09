import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
print("SY-5, Kevin Victor, Roll No.-30")
print("===== SMART HOME (IMPROVED MODEL) =====\n")

# =====================================================
# STEP 1: DOMAIN-SPECIFIC DATASET
# =====================================================
np.random.seed(42)
samples = 800

# Domain labels:
# 0: Living, 1: Kitchen, 2: Bedroom, 3: Washroom, 4: Passage

X = []
y = []

for _ in range(samples):

    domain = np.random.randint(0, 5)

    if domain == 0:  # Living Room
        lighting = np.random.uniform(0.6, 1.0)
        appliance = np.random.uniform(0.2, 0.5)
        comfort = np.random.uniform(0.5, 0.8)

    elif domain == 1:  # Kitchen
        lighting = np.random.uniform(0.4, 0.7)
        appliance = np.random.uniform(0.7, 1.0)
        comfort = np.random.uniform(0.4, 0.6)

    elif domain == 2:  # Bedroom
        lighting = np.random.uniform(0.2, 0.5)
        appliance = np.random.uniform(0.2, 0.4)
        comfort = np.random.uniform(0.8, 1.0)

    elif domain == 3:  # Washroom
        lighting = np.random.uniform(0.5, 0.8)
        appliance = np.random.uniform(0.3, 0.6)
        comfort = np.random.uniform(0.6, 0.9)

    else:  # Passage
        lighting = np.random.uniform(0.6, 1.0)
        appliance = np.random.uniform(0.1, 0.3)
        comfort = np.random.uniform(0.3, 0.6)

    complexity = np.random.uniform(0.2, 1.0)
    energy = np.random.uniform(0.3, 1.0)

    X.append([lighting, appliance, comfort, complexity, energy])
    y.append(domain)

X = np.array(X, dtype=np.float32)
y = np.array(y)

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
# STEP 4: ADVANCED FEATURE ENGINEERING
# =====================================================
def engineer(data):
    l, a, c, comp, e = data.T

    goal_intensity = c + l
    device_score = a * comp
    planning_depth = comp * (1 + c)
    efficiency = e * (1 - comp)

    return np.column_stack([
        goal_intensity,
        device_score,
        planning_depth,
        efficiency
    ])

X_train_f = engineer(X_train_n)
X_test_f = engineer(X_test_n)

# =====================================================
# STEP 5: MODEL
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
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
history = model.fit(
    X_train_f, y_train,
    epochs=35,
    batch_size=16,
    validation_data=(X_test_f, y_test),
    verbose=1
)

# =====================================================
# STEP 8: PLOTS
# =====================================================
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train", "Val"])
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train", "Val"])
plt.show()

# =====================================================
# STEP 9: EVALUATION
# =====================================================
loss, acc = model.evaluate(X_test_f, y_test, verbose=0)

print("\nTest Accuracy:", acc)
print("Test Loss:", loss)
# =====================================================
# STEP 10: COMMAND ENCODER + DOMAIN DETECTION (FIXED)
# =====================================================
def encode_command(cmd):
    cmd = cmd.lower()

    lighting = 1 if "light" in cmd else 0.5
    appliance = 1 if any(x in cmd for x in ["cook", "coffee", "microwave"]) else 0.5
    comfort = 1 if any(x in cmd for x in ["comfort", "cool", "sleep"]) else 0.5
    complexity = 1 if any(x in cmd for x in ["prepare", "arrange"]) else 0.5
    energy = 1 if "energy" in cmd else 0.5

    return np.array([[lighting, appliance, comfort, complexity, energy]], dtype=np.float32)


# =====================================================
# STEP 11: DOMAIN RESOLUTION (NEW)
# =====================================================
def detect_domain_from_command(cmd):
    cmd = cmd.lower()

    if any(x in cmd for x in ["living", "tv", "curtain", "sofa"]):
        return 0  # Living Room

    elif any(x in cmd for x in ["cook", "coffee", "microwave", "kitchen", "sandwich"]):
        return 1  # Kitchen

    elif any(x in cmd for x in ["bed", "sleep", "cool", "bedroom"]):
        return 2  # Bedroom

    elif any(x in cmd for x in ["wash", "water", "bath", "washroom"]):
        return 3  # Washroom

    elif any(x in cmd for x in ["passage", "hall", "corridor"]):
        return 4  # Passage

    return None  # fallback to model


# =====================================================
# STEP 12: BACKWARD PLANNING (FIXED PLANS)
# =====================================================
def generate_plan(domain):

    if domain == 0:  # Living Room
        return [
            "Identify lighting/visual zones",
            "Adjust brightness or curtains",
            "Optimize ambiance for comfort"
        ]

    elif domain == 1:  # Kitchen
        return [
            "Activate required appliances",
            "Prepare ingredients/resources",
            "Execute cooking or preparation sequence"
        ]

    elif domain == 2:  # Bedroom
        return [
            "Adjust room temperature",
            "Prepare bed/cooling system",
            "Optimize environment for rest"
        ]

    elif domain == 3:  # Washroom
        return [
            "Set water temperature",
            "Adjust water flow",
            "Ensure hygiene and comfort conditions"
        ]

    elif domain == 4:  # Passage
        return [
            "Activate motion sensors",
            "Adjust lighting for visibility",
            "Ensure safe navigation"
        ]


# =====================================================
# STEP 13: REAL-TIME SIMULATION (FIXED)
# =====================================================
print("\n===== REAL-TIME SIMULATION (FIXED) =====\n")

commands = [
    "Dim living room lights",
    "Prepare coffee",
    "Pre-cool bedroom",
    "Set washroom water",
    "Activate passage lighting"
]

for t in range(20):
    cmd = commands[t % len(commands)]

    # Step 1: Try rule-based detection
    domain = detect_domain_from_command(cmd)

    # Step 2: If unclear → fallback to ML model
    if domain is None:
        encoded = encode_command(cmd)
        encoded_n = (encoded - mean) / std
        encoded_f = engineer(encoded_n)

        pred = model.predict(encoded_f, verbose=0)
        domain = np.argmax(pred)

    # Step 3: Generate correct plan
    steps = generate_plan(domain)

    print(f"\nTime {t+1}s")
    print("Command:", cmd)
    print("Resolved Domain:", domain)
    print("Plan:")
    for i, s in enumerate(steps, 1):
        print(f"{i}. {s}")
    print("Reason: Domain-aware planning with ML fallback")

    time.sleep(1)