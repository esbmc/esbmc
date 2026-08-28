import numpy as np

a = np.array([1, 3, 5])


def f(x):
    return np.searchsorted(a, x)


r = f(4)
