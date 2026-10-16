import numpy as np

def rows(mat):
    assert mat.shape == (2, 3)
    assert mat.ndim == 2
    assert mat.size == 6
    return len(mat)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = a[0:2]
assert rows(v) == 2
