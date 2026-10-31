import numpy as np


def no_else(c: bool) -> int:
    a = np.array([1, 2])
    assert a[1] == 2
    if c:
        a = np.array([5, 6, 7])
    assert a[0] == (5 if c else 1)
    return len(a)


def both_arms(c: bool) -> float:
    if c:
        a = np.array([1.5, 2.5, 3.5])
    else:
        a = np.array([[1, 2], [3, 4]])
    assert a.ndim == (1 if c else 2)
    assert a.shape[0] == (3 if c else 2)
    return float(a.sum())


assert no_else(True) == 3
assert no_else(False) == 2
assert both_arms(True) == 7.5
assert both_arms(False) == 10.0

g = nondet_bool()
b = np.array([1, 2])
if g:
    b = np.array([1.5, 2.5, 3.5])
assert len(b) == (3 if g else 2)
assert b.sum() == (7.5 if g else 3)
