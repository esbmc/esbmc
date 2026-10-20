import numpy as np

def set_half(row):
    row[0] = 0.5
    return len(row)

r = np.linspace(0.0, 1.0, 5)
w = r[1:3]
m = set_half(w)
assert m == 2
assert r[1] == 0.5
assert r[2] == 0.5
