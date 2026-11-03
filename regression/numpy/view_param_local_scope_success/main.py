import numpy as np

a = np.array([1, 2, 3, 4])
u = a[0:4]

def f(row):
    row[1] = 9
    return len(row)

def g():
    u = a[0:2]
    y = f(u)
    return y

assert g() == 2
assert a[1] == 9
