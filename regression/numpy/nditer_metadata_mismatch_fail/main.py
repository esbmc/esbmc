import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
it = np.nditer(a, flags=["multi_index", "c_index"])
it.iternext()
it.iternext()
it.iternext()
assert it.multi_index == (1, 0)
assert it.index == 2
