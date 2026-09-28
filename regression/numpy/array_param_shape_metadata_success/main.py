import numpy as np


def check_metadata(a):
    # Verify 2-D shape metadata
    shape_0 = a.shape[0]
    shape_1 = a.shape[1]
    ndim = a.ndim
    size = a.size
    len_a = len(a)
    return shape_0, shape_1, ndim, size, len_a


a = np.array([[1, 2, 3], [4, 5, 6]])
s0, s1, nd, sz, ln = check_metadata(a)

assert s0 == 2
assert s1 == 3
assert nd == 2
assert sz == 6
assert ln == 2
