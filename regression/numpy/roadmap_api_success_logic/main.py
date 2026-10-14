import numpy as np

a = np.array([1, 2, 3])

assert np.greater(a, 1) == [False, True, True]
assert np.less_equal(a, 2) == [True, True, False]
assert np.logical_and([True, False], [True, True]) == [True, False]
assert np.logical_not([True, False]) == [False, True]
assert np.where([True, False], [1, 2], [3, 4]) == [1, 4]
