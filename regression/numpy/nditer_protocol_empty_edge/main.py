import numpy as np

z = np.zeros((0,), dtype=int)
it = np.nditer(z, flags=["zerosize_ok"])
assert it.finished
assert not it.iternext()
stopped = False
try:
    next(it)
except StopIteration:
    stopped = True
assert stopped
