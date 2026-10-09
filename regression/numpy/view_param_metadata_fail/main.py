import numpy as np

def check_meta(row):
    l = len(row)
    assert l == 99

a = np.array([1, 2, 3, 4, 5])
v = a[1:4]
check_meta(v)
