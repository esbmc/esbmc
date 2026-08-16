import numpy as np

def f(a):
    return a

x = np.array([[1, 2], [3, 4]])
y = f(x)

assert y[0][0] == 1
assert y[0][1] == 2
assert y[1][0] == 3
assert y[1][1] == 4
