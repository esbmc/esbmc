import numpy as np

calls = [0]


def source():
    calls[0] = calls[0] + 1
    return np.array([1, 2])


try:
    it = np.nditer(source(), op_flags=[["readonly", "readwrite"]])
except ValueError:
    pass
assert calls[0] == 0
