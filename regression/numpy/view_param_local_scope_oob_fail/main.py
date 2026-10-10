import numpy as np

a = np.array([1, 2, 3, 4])
u = a[0:4]

def f(row):
    row[3] = 0
    return len(row)

def g():
    u = a[0:2]
    y = f(u)
    return y

r = g()
