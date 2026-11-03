import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

a = np.array([1, 2, 3, 4])
f = np.zeros(4)
v = a[0:2]
w = f[0:2]
n = set_first(v)
m = set_first(w)
