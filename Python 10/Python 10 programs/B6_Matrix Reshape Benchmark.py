import random
from time import perf_counter
import tensorflow as tf
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# SETTINGS
# -------------------------------
ROWS, COLS = 500, 500
NEW_ROWS, NEW_COLS = 250, 1000

# -------------------------------
# GENERATE ORIGINAL MATRIX
# -------------------------------
def generate_matrix(r, c):
    return [[random.randint(1, 9) for _ in range(c)] for _ in range(r)]

original_matrix = generate_matrix(ROWS, COLS)

# -------------------------------
# PRINT FUNCTION
# -------------------------------
def print_matrix(mat):
    for row in mat:
        print(row)

print("Original Matrix ({}x{}):".format(ROWS, COLS))
print_matrix(original_matrix)

# -------------------------------
# MANUAL RESHAPE
# -------------------------------
def manual_reshape(matrix, new_r, new_c):
    # Flatten the matrix
    flat = []
    for row in matrix:
        for val in row:
            flat.append(val)

    # Check if reshape is possible
    if len(flat) != new_r * new_c:
        raise ValueError("Reshape not possible!")

    # Build reshaped matrix
    reshaped = []
    index = 0

    for i in range(new_r):
        row = []
        for j in range(new_c):
            row.append(flat[index])
            index += 1
        reshaped.append(row)

    return reshaped

# Timing manual reshape
start_manual = perf_counter()
manual_result = manual_reshape(original_matrix, NEW_ROWS, NEW_COLS)
end_manual = perf_counter()

manual_time = (end_manual - start_manual) * 1_000_000

# -------------------------------
# TENSORFLOW RESHAPE
# -------------------------------
tensor = tf.constant(original_matrix)

start_tf = perf_counter()
tf_result = tf.reshape(tensor, (NEW_ROWS, NEW_COLS))
tf_result_np = tf_result.numpy()  # Force execution
end_tf = perf_counter()

tf_time = (end_tf - start_tf) * 1_000_000

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\nManual Reshaped Matrix ({}x{}):".format(NEW_ROWS, NEW_COLS))
print_matrix(manual_result)

print("\nTensorFlow Reshaped Matrix ({}x{}):".format(NEW_ROWS, NEW_COLS))
print(tf_result_np)

# -------------------------------
# PERFORMANCE
# -------------------------------
print("\n===== PERFORMANCE =====")
print("Manual Reshape Time: {:.2f} microseconds".format(manual_time))
print("TensorFlow Reshape Time: {:.2f} microseconds".format(tf_time))

# -------------------------------
# CORRECTNESS CHECK
# -------------------------------
match = True
for i in range(NEW_ROWS):
    for j in range(NEW_COLS):
        if manual_result[i][j] != tf_result_np[i][j]:
            match = False
            break

print("\nResults Match:", match)

# -------------------------------
# SPEED COMPARISON
# -------------------------------
if tf_time > 0:
    print("Speedup (Manual / TF): {:.2f}x".format(manual_time / tf_time))
else:
    print("TensorFlow time too small to measure accurately")