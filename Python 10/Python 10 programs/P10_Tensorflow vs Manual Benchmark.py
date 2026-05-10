import tensorflow as tf
from time import perf_counter
import random
print("SY-5, Kevin Victor, Roll No.-30")
print("===== TENSOR OPERATIONS: MANUAL vs TENSORFLOW =====\n")

# =====================================================
# PART 1: CREATE TENSORS OF DIFFERENT RANKS
# =====================================================

# -------- Manual (Python) --------
print("===== MANUAL (PYTHON) =====\n")

# Scalar (Rank 0)
scalar_py = 10

# Vector (Rank 1)
vector_py = [random.randint(1, 10) for _ in range(1000)]

# Matrix (Rank 2)
matrix_py = [[random.randint(1, 10) for _ in range(100)] for _ in range(100)]

# 3D Tensor (Rank 3)
tensor3d_py = [[[random.randint(1, 5) for _ in range(20)] for _ in range(20)] for _ in range(10)]

print("# Scalar:", scalar_py)
print("# Vector length:", len(vector_py))
print("# Matrix size:", len(matrix_py), "x", len(matrix_py[0]))
print("# 3D Tensor shape:", len(tensor3d_py), "x", len(tensor3d_py[0]), "x", len(tensor3d_py[0][0]))


# -------- TensorFlow --------
print("\n===== TENSORFLOW =====\n")

scalar_tf = tf.constant(10, dtype=tf.int32)
vector_tf = tf.constant(vector_py, dtype=tf.int32)
matrix_tf = tf.constant(matrix_py, dtype=tf.float32)
tensor3d_tf = tf.constant(tensor3d_py, dtype=tf.float32)

print("# Scalar:", scalar_tf.numpy())
print("# Vector shape:", vector_tf.shape)
print("# Matrix shape:", matrix_tf.shape)
print("# 3D Tensor shape:", tensor3d_tf.shape)

# =====================================================
# PART 2: BASIC OPERATIONS (LARGE SCALE)
# =====================================================

print("\n===== BASIC OPERATIONS =====\n")

# -------- Manual Operations --------
start_manual = perf_counter()

# Addition
vector_add_py = [a + b for a, b in zip(vector_py, vector_py)]

# Subtraction
vector_sub_py = [a - b for a, b in zip(vector_py, vector_py)]

# Multiplication (element-wise)
vector_mul_py = [a * b for a, b in zip(vector_py, vector_py)]

# Matrix Multiplication (heavy)
def manual_matmul(A, B):
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

matrix_mul_py = manual_matmul(matrix_py, matrix_py)

end_manual = perf_counter()
manual_time = (end_manual - start_manual) * 1_000_000


# -------- TensorFlow Operations --------
start_tf = perf_counter()

# Addition
vector_add_tf = tf.add(vector_tf, vector_tf)

# Subtraction
vector_sub_tf = tf.subtract(vector_tf, vector_tf)

# Multiplication
vector_mul_tf = tf.multiply(vector_tf, vector_tf)

# Matrix Multiplication
matrix_mul_tf = tf.matmul(matrix_tf, matrix_tf)

# Force execution
_ = matrix_mul_tf.numpy()

end_tf = perf_counter()
tf_time = (end_tf - start_tf) * 1_000_000


# =====================================================
# PART 3: TENSOR ATTRIBUTES
# =====================================================

print("\n===== TENSOR ATTRIBUTES =====\n")

print("# Scalar → Shape:", scalar_tf.shape,
      "| Rank:", tf.rank(scalar_tf).numpy(),
      "| Dtype:", scalar_tf.dtype)

print("# Vector → Shape:", vector_tf.shape,
      "| Rank:", tf.rank(vector_tf).numpy(),
      "| Dtype:", vector_tf.dtype)

print("# Matrix → Shape:", matrix_tf.shape,
      "| Rank:", tf.rank(matrix_tf).numpy(),
      "| Dtype:", matrix_tf.dtype)

print("# 3D Tensor → Shape:", tensor3d_tf.shape,
      "| Rank:", tf.rank(tensor3d_tf).numpy(),
      "| Dtype:", tensor3d_tf.dtype)


# =====================================================
# PART 4: RESULTS
# =====================================================

print("\n===== PERFORMANCE COMPARISON =====\n")

print("Manual Operations Time: {:.2f} microseconds".format(manual_time))
print("TensorFlow Operations Time: {:.2f} microseconds".format(tf_time))

if tf_time > 0:
    print("Speedup (on avg) (Manual / TensorFlow): {:.2f}x".format(manual_time / tf_time))


# =====================================================
# PART 5: SAMPLE OUTPUT CHECK
# =====================================================

print("\n===== SAMPLE OUTPUT CHECK =====\n")

print("# Manual Vector Add (first 5):", vector_add_py[:5])
print("# TF Vector Add (first 5):", vector_add_tf.numpy()[:5])

print("\n# Manual Matrix Mul (first row first 5):", matrix_mul_py[0][:5])
print("# TF Matrix Mul (first row first 5):", matrix_mul_tf.numpy()[0][:5])


# =====================================================
# FINAL NOTES
# =====================================================

print("\n===== NOTES =====\n")

print("# Manual operations use Python loops → slower for large data")
print("# TensorFlow uses optimized backend (C/C++/parallelism)")
print("# Matrix multiplication highlights major performance difference")
print("# TensorFlow handles large-scale computation efficiently")