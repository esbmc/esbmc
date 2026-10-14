import numpy as np


def alias_preserve_shape(a):
    b = a  # Create alias
    return b[1, 1]


a = np.array([[1, 2], [3, 4]])
result = alias_preserve_shape(a)

# b[1, 1] should be 4
assert result == 4
