import numpy as np

a = np.array([3, 1, 2])
a.sort(0)

# Wrong assertion to verify the test detects incorrect values
assert a[0] == 2  # Should be 1
