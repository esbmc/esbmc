import numpy as np

def apply_fn(row, fn):
    return fn(row)

a = np.array([2, 4, 6])
v = a[0:]
result = apply_fn(v, lambda x: x[0])
assert result == 2
