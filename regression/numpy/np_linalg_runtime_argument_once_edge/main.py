import numpy as np

calls = 0


def make_a():
    global calls
    calls = calls * 10 + 1
    return np.array([[2.0, 1.0], [1.0, 3.0]])


def make_b():
    global calls
    calls = calls * 10 + 2
    return np.array([3.0, 5.0])


def make_row():
    global calls
    calls = calls * 10 + 3
    return np.array([[1.0, 2.0, 3.0]])


x = np.linalg.solve(make_a(), make_b())
assert calls == 12
assert abs(x[0] - 0.8) < 1e-9 and abs(x[1] - 1.4) < 1e-9
try:
    b = np.linalg.inv(make_row())
except np.linalg.LinAlgError:
    calls = calls * 10 + 4
assert calls == 1234
