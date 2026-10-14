import numpy as np


def f(cond):
    if cond:
        return np.array([1, 2, 3])
    else:
        return "hello"


c = nondet_bool()
y = f(c)
assert y[0] == 1
