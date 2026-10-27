import numpy as np

calls = [0]


def source():
    calls[0] = calls[0] + 1
    if calls[0] > 1:
        raise ValueError("evaluated twice")
    return np.array([1, 2])


caught = False
try:
    it = np.nditer(source(), op_flags=[["readonly", "readwrite"]])
except ValueError:
    caught = True
assert caught
assert calls[0] == 1

a = np.array([5, 6])
it = np.nditer(a)
a = np.array([0, 0])
assert next(it) == 5
