import numpy as np

g = None


def leak(a):
    global g
    g = a[0]


x = np.array([[1, 2], [3, 4]])
leak(x)
g[0] = 999

assert x[0][0] == 1
