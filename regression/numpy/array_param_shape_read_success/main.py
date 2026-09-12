import numpy as np


def read_2d(a):
    # Read 2-D indices in multiple forms
    v1 = a[1][0]
    v2 = a[0, 1]
    v3 = a[1, 1]
    return v1 + v2 + v3


a = np.array([[1, 2], [3, 4]])
result = read_2d(a)

# Expected: 3 + 2 + 4 = 9
assert result == 9
