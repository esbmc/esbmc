import numpy as np


assert np.arange(5, -1, -2) == [5, 3, 1]
assert np.arange(0) == []
assert np.full(0, 7) == []
assert np.full((1, 3), -0.0) == [[-0.0, -0.0, -0.0]]
assert np.eye(3, 2) == [[1, 0], [0, 1], [0, 0]]
assert np.linspace(0.0, 1.0, 1) == [0.0]
assert np.greater_equal([1, 2], [1, 3]) == [True, False]
assert np.where(False, [1, 2], [3, 4]) == [3, 4]
assert np.argmin([2, 2, 1]) == 2
assert np.argmax([2, 2, 1]) == 0
assert np.logical_or([False, True], [True, False]) == [True, True]
assert np.logical_not(False) == True
