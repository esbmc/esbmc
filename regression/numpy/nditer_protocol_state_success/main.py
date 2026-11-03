import numpy as np

a = np.array([[1, 2], [3, 4]])
it = np.nditer(a)
assert it.shape == (4,)
assert it.ndim == 1
assert it.itersize == 4
assert not it.finished
assert it.iterindex == 0

steps = 1
while it.iternext():
    steps += 1
assert steps == 4
assert it.finished

m = np.nditer(a, flags=["multi_index"])
assert m.shape == (2, 2)
assert m.ndim == 2

first = np.nditer(a)
moved = first.iternext()
assert moved
assert first.iterindex == 1
