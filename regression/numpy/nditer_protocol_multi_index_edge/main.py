import numpy as np

a = np.array([[1, 2], [3, 4]])
it = np.nditer(a, flags=["multi_index"])
next(it)
next(it)
assert it.multi_index == (0, 1)
assert it.iterindex == 1
while it.iternext():
    pass
caught = False
try:
    m = it.multi_index
except ValueError:
    caught = True
assert caught
