import numpy as np

calls = [0]


def source():
    calls[0] = calls[0] + 1
    return np.array([1, 2, 3])


total = 0
for x in np.nditer(source()):
    total += int(x)
assert total == 6
assert calls[0] == 1
