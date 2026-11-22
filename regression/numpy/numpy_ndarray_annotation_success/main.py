import numpy as np


def values() -> np.ndarray:
    return np.array([8.0, 27.0])


def scaled(a: np.ndarray) -> np.ndarray:
    return a * 2.0


r = np.cbrt(values())
assert r[0] > 1.9999 and r[0] < 2.0001
s = scaled(values())
assert s[1] == 54.0
