import numpy as np

a = np.array([2, 1, 2, 1, 2])
result = np.sort(a, kind="stable")

assert result[0] == 1
assert result[1] == 1
assert result[2] == 2
assert result[3] == 2
assert result[4] == 2
