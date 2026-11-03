import numpy as np


def grid(n: int):
    return np.linspace(0, 1, n)


a = grid(3)
b = grid(4)
assert len(a) == 3 and len(b) == 4
