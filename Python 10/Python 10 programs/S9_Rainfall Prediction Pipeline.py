import tensorflow as tf
print("SY-5, Kevin Victor, Roll No.-30")
print("\n===== WEATHER ML PIPELINE USING TENSORFLOW =====\n")

# -------------------------------
# STEP 1: RAW DATA (REALISTIC)
# -------------------------------
# 5 days sample (features per day)

# Features:
# [temp (°C), humidity (%), pressure (hPa), ocean_temp (°C),
#  wind_speed (km/h), instability_index]

weather_data = [
    [36.5, 65, 1008, 28.5, 12, 0.6],
    [37.2, 60, 1006, 29.0, 15, 0.7],
    [35.8, 72, 1004, 28.8, 18, 0.8],
    [34.9, 80, 1002, 28.2, 20, 0.9],
    [33.5, 85, 1000, 27.9, 22, 0.95]
]

print("# Raw Weather Data:")
print(weather_data)

# -------------------------------
# STEP 2: CONVERT TO TENSOR
# -------------------------------
weather_tensor = tf.constant(weather_data, dtype=tf.float32)

print("\n# Tensor Form:")
print(weather_tensor)

print("\n# Shape:", weather_tensor.shape)

# -------------------------------
# STEP 3: NORMALIZATION
# -------------------------------
# Normalize values for ML model

mean = tf.reduce_mean(weather_tensor, axis=0)
std = tf.math.reduce_std(weather_tensor, axis=0)

normalized_data = (weather_tensor - mean) / std

print("\n===== NORMALIZED DATA =====")
print(normalized_data)

# -------------------------------
# STEP 4: FEATURE EXTRACTION
# -------------------------------
temp = normalized_data[:, 0]
humidity = normalized_data[:, 1]
pressure = normalized_data[:, 2]
ocean_temp = normalized_data[:, 3]
wind = normalized_data[:, 4]
instability = normalized_data[:, 5]

print("\n===== FEATURE EXTRACTION =====")
print("Temperature:", temp)
print("Humidity:", humidity)
print("Pressure:", pressure)

# -------------------------------
# STEP 5: SIMPLE RAIN MODEL (LOGIC-BASED)
# -------------------------------
# Weighted combination (simulating ML model)

rain_score = (
    0.3 * humidity +
    0.2 * (-pressure) +   # low pressure favors rain
    0.2 * ocean_temp +
    0.2 * instability +
    0.1 * wind
)

print("\n===== RAIN SCORE =====")
print(rain_score)

# -------------------------------
# STEP 6: PREDICTION
# -------------------------------
rain_prediction = tf.where(rain_score > 0.5, 1, 0)

print("\n===== RAIN PREDICTION =====")
print(rain_prediction.numpy())

# -------------------------------
# STEP 7: INTERPRETATION
# -------------------------------
print("\n===== WEATHER REPORT =====\n")

for i in range(len(weather_data)):
    print(f"Day {i+1}:")

    print("  Rain Prediction:", "YES" if rain_prediction[i] == 1 else "NO")

    # Reasoning
    reasons = []

    if weather_data[i][1] > 70:
        reasons.append("High Humidity")

    if weather_data[i][2] < 1005:
        reasons.append("Low Pressure")

    if weather_data[i][5] > 0.8:
        reasons.append("High Atmospheric Instability")

    if weather_data[i][4] > 18:
        reasons.append("Strong Winds")

    if weather_data[i][3] > 28:
        reasons.append("Warm Ocean (Moisture Source)")

    print("  Reasons:", ", ".join(reasons) if reasons else "Stable Conditions")
    print()