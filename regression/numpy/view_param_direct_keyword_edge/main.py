import numpy as np

def compute(v, scale):
    return v[0] * scale

a = np.array([5, 6, 7])
result = compute(v=a[0:], scale=4)
assert result == 20
