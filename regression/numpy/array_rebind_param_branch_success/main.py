import numpy as np


def f(a, c: bool) -> int:
    v = a[1:]
    if c:
        a = np.zeros(3)
    return len(a) + len(v)


assert f(np.zeros(3), True) == 5
assert f(np.zeros(3), False) == 5
