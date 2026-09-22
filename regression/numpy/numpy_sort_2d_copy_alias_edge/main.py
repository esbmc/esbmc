import numpy as np

a = np.array([[3, 1], [4, 2]])
result = np.sort(a, axis=1)

# Mutate result - shouldn't affect origin
result[0, 0] = 999
assert a[0, 0] == 3  # Original unchanged
