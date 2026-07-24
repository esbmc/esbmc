import numpy as np

g = 0


def bump():
    global g
    g = 1
    return 0


def passthrough(a, unused):
    return a


x = np.array([[1, 2], [3, 4]])
passthrough(x, bump())

assert g == 0
