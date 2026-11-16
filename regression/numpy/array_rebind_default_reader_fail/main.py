import numpy as np

a = np.array([1, 2])


def f(x=a) -> int:
    return len(x) + len(a)


a = np.array([5, 6, 7])
assert f() == 5
