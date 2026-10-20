import numpy as np

a = np.array([1, 2])


def first() -> int:
    return int(a[0])


assert a[1] != 2
a = np.array([5, 6, 7])
assert first() == 5
