import numpy as np

def set_half(row):
    row[0] = 0.5
    return len(row)

f = np.arange(0.0, 4.0)
u = f[1:3]
k = set_half(u)
assert f[1] == 1.0
