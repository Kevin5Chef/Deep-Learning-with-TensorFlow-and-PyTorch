import tensorflow as tf
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# REALISTIC DATASET (Pune Max Temp - Last 20 Days)
# -------------------------------
# Values based on realistic April range (33°C to 40°C)

max_temp_1d = [
    33.7, 34.5, 35.2, 36.1, 37.0,
    38.2, 39.1, 40.0, 38.5, 37.8,
    36.9, 35.5, 34.8, 33.9, 34.2,
    35.7, 36.8, 37.6, 38.9, 39.5
]

print("===== ORIGINAL DATA (1D Tensor) =====\n")

# -------------------------------
# CREATE 1D TENSOR
# -------------------------------
temp_tensor_1d = tf.constant(max_temp_1d)

print("1D Tensor:")
print(temp_tensor_1d.numpy())

print("\nProperties:")
print("Shape:", temp_tensor_1d.shape)
print("Rank:", tf.rank(temp_tensor_1d).numpy())

# -------------------------------
# RESHAPE TO 2D (Model Input Format)
# -------------------------------
print("\n===== RESHAPED DATA (2D Tensor) =====\n")

# Reshape → (20,1) → column format
temp_tensor_2d = tf.reshape(temp_tensor_1d, (20, 1))

print("2D Tensor:")
print(temp_tensor_2d.numpy())

print("\nProperties:")
print("Shape:", temp_tensor_2d.shape)
print("Rank:", tf.rank(temp_tensor_2d).numpy())

# -------------------------------
# EXPLANATION
# -------------------------------
print("\n===== WHY RESHAPING IS REQUIRED =====\n")

print("# Machine learning models expect input in 2D format:")
print("# (samples, features)")

print("\n# In this case:")
print("# 1D Tensor → shape (20,) → NOT suitable for model input")
print("# 2D Tensor → shape (20,1) → 20 samples, 1 feature")

print("\n# Reshaping organizes data without changing values")
print("# tf.reshape only changes the structure (shape), not the data itself")