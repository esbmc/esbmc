import numpy as np


def after(n: int) -> float:
    a = np.array([1, 2])
    for i in range(n):
        assert a[0] == (1 if i == 0 else 5.5)
        a = np.array([5.5, 6.5, 7.5])
    assert a.ndim == 1
    assert len(a) == (3 if n > 0 else 2)
    return float(a.sum())


assert after(0) == 3.0
assert after(1) == 19.5
assert after(3) == 19.5
