
import tensorflow as tf
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# REALISTIC DATASET
# -------------------------------

# Scalar → One temperature value (latest day)
temperature_scalar = 36.5  # °C

# Vector → Daily AC sales + revenue (last 15 days)
# Format: [units_sold, revenue_in_rupees]
daily_sales = [
    [12, 480000], [15, 600000], [18, 720000],
    [10, 400000], [20, 800000], [22, 880000],
    [25, 1000000], [19, 760000], [17, 680000],
    [23, 920000], [26, 1040000], [30, 1200000],
    [28, 1120000], [24, 960000], [21, 840000]
]

# Matrix → Student marks (OS, AI, Python)
student_marks = [
    [78, 85, 90],
    [88, 79, 92],
    [67, 74, 80],
    [90, 91, 89],
    [76, 84, 88]
]

# -------------------------------
# TENSORFLOW IMPLEMENTATION
# -------------------------------
print("===== TENSORFLOW IMPLEMENTATION =====\n")



# Create tensors
tf_scalar = tf.constant(temperature_scalar)
tf_vector = tf.constant(daily_sales)
tf_matrix = tf.constant(student_marks)

# Force execution
_ = tf_matrix.numpy()

# -------------------------------
# DISPLAY TENSORS
# -------------------------------
print("# Scalar Tensor (Temperature):")
print(tf_scalar.numpy())

print("\n# Vector Tensor (Daily Sales - Units & Revenue):")
print(tf_vector.numpy())

print("\n# Matrix Tensor (Student Marks - OS, AI, Python):")
print(tf_matrix.numpy())

# -------------------------------
# TENSOR PROPERTIES
# -------------------------------
print("\n===== TENSOR PROPERTIES =====\n")

print("# Scalar:")
print("Shape:", tf_scalar.shape)
print("Rank:", tf.rank(tf_scalar).numpy())
print("Data Type:", tf_scalar.dtype)

print("\n# Vector:")
print("Shape:", tf_vector.shape)
print("Rank:", tf.rank(tf_vector).numpy())
print("Data Type:", tf_vector.dtype)

print("\n# Matrix:")
print("Shape:", tf_matrix.shape)
print("Rank:", tf.rank(tf_matrix).numpy())
print("Data Type:", tf_matrix.dtype)


# -------------------------------
# FINAL NOTES
# -------------------------------
print("\n===== NOTES =====\n")

print("# This dataset is NOT internally related — used purely for demonstration")
print("# Scalar → Single value (temperature)")
print("# Vector → Sequence of records (daily AC sales & revenue)")
print("# Matrix → Tabular data (student marks across subjects)")
print("# TensorFlow focuses on structure (shape, rank), not meaning of data")