import numpy as np


assert np.sum([1, 2, 3]) == 6
assert np.prod([2, 3]) == 6
assert np.min([3, 2, 1]) == 1
assert np.max([1, 5, 2]) == 5
assert np.where([True, False], [1, 2], [3, 4]) == [1, 4]

assert np.arange(4) == [0, 1, 2, 4]
