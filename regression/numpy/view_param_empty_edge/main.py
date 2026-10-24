import numpy as np

def check_empty(row):
    l = len(row)
    assert l == 0

a = np.array([1, 2, 3])
v = a[1:1]
check_empty(v)
