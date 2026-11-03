import numpy as np

def outer(row):
    row[1] = 0
    k = inner(row)
    return k

def inner(row: int):
    return 1

a = np.array([1, 2, 3, 4])
v = a[0:]
x = outer(v)
