import numpy as np


def branch_incompatible(x):
    if x > 0:
        return np.array([1, 2])  # 2 elements
    else:
        return np.array([1, 2, 3])  # 3 elements - incompatible!


result = branch_incompatible(1)
assert result[0] == 1
