import numpy as np

def sum_first_col(mat):
    t = mat.T
    return t[0][0] + t[0][1]

a = np.array([[1, 2], [3, 4]])
v = a[0:]
result = sum_first_col(v)
assert result == 4
