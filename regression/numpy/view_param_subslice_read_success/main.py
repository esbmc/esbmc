import numpy as np

def second(row):
    sub = row[1:]
    return sub[0]

a = np.array([10, 20, 30])
v = a[0:]
assert second(v) == 20
