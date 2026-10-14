import numpy as np


def make():
    a = np.array([3, 1, 2])
    return a


result = make()
sorted_result = np.sort(result, kind="stable")
assert sorted_result[0] == 1
assert sorted_result[1] == 2
assert sorted_result[2] == 3
