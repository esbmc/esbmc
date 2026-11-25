import numpy as np

calls = 0


def make_a():
    global calls
    calls = calls * 10 + 1
    return np.array([[1.0, 0.0, 0.0]])


def make_b():
    global calls
    calls = calls * 10 + 2
    return np.array([1.0])


try:
    x = np.linalg.solve(make_a(), make_b())
except np.linalg.LinAlgError:
    calls = calls * 10 + 3
assert calls == 123
try:
    y = np.linalg.solve([[1.0, 0.0, 0.0]], make_b())
except np.linalg.LinAlgError:
    calls = calls * 10 + 3
assert calls == 12323
