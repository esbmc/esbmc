import numpy as np


def f(cond):
    if cond:
        return np.array([1, 2, 3])
    else:
        return "hello"


y = f(True)
assert y[0] == 1
