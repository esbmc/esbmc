import numpy as np

def read_kw(v, scale):
    return v[0] * scale

a = np.array([10, 20, 30])
view = a[0:]
result = read_kw(v=view, scale=2)
assert result == 20
