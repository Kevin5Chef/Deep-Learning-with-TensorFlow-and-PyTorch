from time import perf_counter
import tensorflow as tf
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# MANUAL (PYTHON) IMPLEMENTATION
# -------------------------------

print("===== MANUAL (PYTHON) =====\n")

# Constant (immutable concept)
manual_const = 10

# Variable (mutable)
manual_var = 10

print("# Manual Constant initial value:", manual_const)
print("# Manual Variable initial value:", manual_var)

# Timing manual reassignment
start_manual = perf_counter()

# Reassign variable
manual_var = 20

# Try modifying constant (Python allows it, but conceptually it's not a constant)
manual_const = 20  # No restriction in Python

end_manual = perf_counter()

manual_time = (end_manual - start_manual) * 1_000_000

print("\n# After reassignment:")
print("# Manual Variable updated to:", manual_var)
print("# Manual Constant ALSO changed (Python doesn't enforce constants):", manual_const)

print("\n# NOTE: Python does NOT enforce true constants — both can change!\n")


# -------------------------------
# TENSORFLOW IMPLEMENTATION
# -------------------------------

print("===== TENSORFLOW =====\n")

# TensorFlow Constant (immutable)
tf_const = tf.constant(10)

# TensorFlow Variable (mutable)
tf_var = tf.Variable(10)

print("# TF Constant initial value:", tf_const.numpy())
print("# TF Variable initial value:", tf_var.numpy())

# Timing TensorFlow assignment
start_tf = perf_counter()

# Assign new value to variable
tf_var.assign(20)

# Attempt to change constant (not allowed)
try:
    tf_const.assign(20)  # This will raise an error
except Exception as e:
    tf_const_error = str(e)

end_tf = perf_counter()

tf_time = (end_tf - start_tf) * 1_000_000

print("\n# After assignment:")
print("# TF Variable updated to:", tf_var.numpy())
print("# TF Constant remains unchanged:", tf_const.numpy())

print("\n# Attempting to modify TF Constant gives error:")
print("#", tf_const_error)


# -------------------------------
# PERFORMANCE COMPARISON
# -------------------------------

print("\n===== PERFORMANCE =====\n")

print("Manual Reassignment Time: {:.2f} microseconds".format(manual_time))
print("TensorFlow Assignment Time: {:.2f} microseconds".format(tf_time))

if tf_time > 0:
    print("Speed Ratio (Manual / TF): {:.2f}x".format(manual_time / tf_time))
else:
    print("TensorFlow time too small to measure accurately")

print("\nTensorflow's poor performance here is due to the overhead of creating and managing tensors, \nwhich is not optimized for simple scalar operations. \nIn contrast, Python's variable assignment is extremely fast for simple data types.")

# -------------------------------
# FINAL SUMMARY
# -------------------------------

print("\n===== SUMMARY =====\n")

print("# Python:")
print("# - No strict constants")
print("# - Variables and 'constants' behave the same")

print("\n# TensorFlow:")
print("# - tf.Variable is mutable (can change using assign())")
print("# - tf.constant is immutable (cannot be changed)")