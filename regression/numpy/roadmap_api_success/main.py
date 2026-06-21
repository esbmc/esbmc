import numpy as np


a = np.array([1, 2, 3])

assert np.sum(a) == 6
assert np.prod(a) == 6
assert np.min(a) == 1
assert np.max(a) == 3
assert np.mean(a) == 2.0
assert np.argmin(a) == 0
assert np.argmax(a) == 2

assert np.greater(a, 1) == [False, True, True]
assert np.less_equal(a, 2) == [True, True, False]
assert np.logical_and([True, False], [True, True]) == [True, False]
assert np.logical_not([True, False]) == [False, True]

assert np.where([True, False], [1, 2], [3, 4]) == [1, 4]

assert np.arange(5) == [0, 1, 2, 3, 4]
assert np.full((2, 2), 7) == [[7, 7], [7, 7]]
assert np.eye(2) == [[1, 0], [0, 1]]
assert np.identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
assert np.linspace(0.0, 1.0, 3) == [0.0, 0.5, 1.0]
