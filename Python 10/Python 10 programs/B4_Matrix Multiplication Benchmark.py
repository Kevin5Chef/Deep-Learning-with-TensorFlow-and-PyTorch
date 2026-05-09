import random
from time import perf_counter
import tensorflow as tf
print("SY-5, Kevin Victor, Roll No.-30")
# Matrix size
SIZE = 300

# Generate random matrices
def generate_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]

A = generate_matrix(SIZE)
B = generate_matrix(SIZE)

# -------------------------------
# MANUAL MATRIX MULTIPLICATION
# -------------------------------
def manual_matrix_multiply(A, B):
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Timing manual method
start_manual = perf_counter()
manual_result = manual_matrix_multiply(A, B)
end_manual = perf_counter()

manual_time = (end_manual - start_manual) * 1_000_000  # convert to microseconds

# -------------------------------
# TENSORFLOW MATRIX MULTIPLICATION
# -------------------------------
A_tf = tf.constant(A, dtype=tf.float32)
B_tf = tf.constant(B, dtype=tf.float32)

start_tf = perf_counter()
tf_result = tf.matmul(A_tf, B_tf)
end_tf = perf_counter()

tf_time = (end_tf - start_tf) * 1_000_000  # convert to microseconds

# -------------------------------
# RESULTS
# -------------------------------
print("Manual Multiplication Time: {:.2f} microseconds".format(manual_time))
print("TensorFlow Multiplication Time: {:.2f} microseconds".format(tf_time))

# Convert TensorFlow result to Python list for comparison
tf_result_list = tf_result.numpy()

# Check small portion for correctness
print("\nSample comparison (first 3x3 block):")
for i in range(3):
    for j in range(3):
        print("Manual:", round(manual_result[i][j], 4),
              "TF:", round(tf_result_list[i][j], 4))
    print()

# Speed comparison
if tf_time > 0:
    speedup = manual_time / tf_time
    print("TensorFlow is approximately {:.2f}x faster".format(speedup))
else:
    print("TensorFlow execution time too small to measure accurately")