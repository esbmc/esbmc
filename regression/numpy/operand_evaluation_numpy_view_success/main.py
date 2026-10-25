import numpy as np

a = np.array([1, 2, 3])


def poke() -> int:
    a[0] = 9
    return 1


x = a[0] + poke()
assert x == 2
assert a[0] == 9
