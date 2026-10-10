import numpy as np

def compute(scale, v):
    return v[0] * scale

a = np.array([10, 20, 30])
view = a[0:]
result = compute(v=view, scale=3)
assert result == 30
