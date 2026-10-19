import numpy as np


def make():
    return np.linspace(0, 1, 3, dtype=int)


a = make()
assert a[1] == 0.5
