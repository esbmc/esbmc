import numpy as np

one = np.array([42])
it = np.nditer(one)
assert it.value == 42
assert not it.iternext()
assert it.finished
assert it.itersize == 1
caught = False
try:
    s = it.shape
except ValueError:
    caught = True
assert caught
