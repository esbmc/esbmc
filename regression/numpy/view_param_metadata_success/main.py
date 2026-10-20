import numpy as np

def check_meta(row):
    l = len(row)
    s = row.shape[0]
    nd = row.ndim
    sz = row.size
    assert l == 3
    assert s == 3
    assert nd == 1
    assert sz == 3

a = np.array([1, 2, 3, 4, 5])
v = a[1:4]
check_meta(v)
