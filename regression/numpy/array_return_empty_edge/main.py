import numpy as np

def f(a):
    return a

x = np.array([])
y = f(x)

assert y.shape[0] == 0
