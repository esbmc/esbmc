import numpy as np

calls = 0


def make():
    global calls
    calls += 1
    return np.array([[2.0, 1.0], [1.0, 3.0]])


d = np.linalg.det(make())
assert calls == 1
assert abs(d - 5.0) < 1e-9
