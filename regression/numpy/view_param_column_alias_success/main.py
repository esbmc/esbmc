import numpy as np

def zero_first(col):
    old = col[0]
    col[0] = 0
    return old

a = np.array([[1, 2], [3, 4]])
v = a[:, 0]
result = zero_first(v)
assert result == 1
assert a[0][0] == 0
