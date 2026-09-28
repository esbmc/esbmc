import numpy as np

a = np.array([3, 1, 2])
result = np.sort(a, kind="stable")

assert result[0] == 1
assert result[1] == 2
assert result[2] == 3
