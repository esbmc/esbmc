import numpy as np


def pick_row(a, take_second):
    if take_second:
        return a[1]
    return a[0]


x = np.array([[1, 2], [3, 4]])
first = pick_row(x, False)
second = pick_row(x, True)

assert first[0] == 1
assert first[1] == 2
assert second[0] == 3
assert second[1] == 4
