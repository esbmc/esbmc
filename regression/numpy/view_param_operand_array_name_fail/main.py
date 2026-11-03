import numpy as np

def passthrough(row):
    row[0] = 42
    return row

a = np.array([1, 2, 3, 4])
b = np.array([5, 6])
v = a[0:2]
z = b + passthrough(v)[0]
