import numpy as np

calls = 0


def arg() -> float:
    global calls
    calls += 1
    return 0.5


def values():
    global calls
    calls += 1
    return np.array([0.5, 1.0])


a = np.isnan(arg())
b = np.isfinite(arg())
c = np.cbrt(values())
d = np.isinf(values())
assert calls == 4
assert not a and b and not d[0]
