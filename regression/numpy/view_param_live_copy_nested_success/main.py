import numpy as np

def bump(v):
    v[0] = v[0] + 100

def outer(row):
    sub = row[1:]
    saved = sub.copy()
    bump(sub)
    return saved[0] + sub[0]

a = np.array([10, 20, 30])
v = a[0:]
assert outer(v) == 140
assert a[1] == 120
