import numpy as np


def f(c: bool) -> int:
    a = np.array([1, 2])
    assert a[1] != 2
    if c:
        a = np.array([5, 6, 7])
    return len(a)


f(True)
