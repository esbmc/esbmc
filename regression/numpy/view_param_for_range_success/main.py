import numpy as np

def fill(row):
    for i in range(3):
        row[i] = i * 2

a = np.array([0, 0, 0])
v = a[0:]
fill(v)
assert a[0] == 0
assert a[1] == 2
assert a[2] == 4
