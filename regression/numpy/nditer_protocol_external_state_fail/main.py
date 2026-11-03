import numpy as np


def consume(i):
    return next(i)


a = np.array([5, 6, 7])
it = np.nditer(a)
assert consume(it) == 5
