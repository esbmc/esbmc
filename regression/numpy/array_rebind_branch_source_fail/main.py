import numpy as np


def f(c: bool) -> int:
    a = np.array([1, 2])
    b = a
    assert b[1] != 2
    if c:
        a = np.array([5, 6])
    return int(b[0])


f(True)
