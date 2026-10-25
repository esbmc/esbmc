import numpy as np

a = np.array([[1, 2], [3, 4]])
it = np.nditer(a, flags=["multi_index", "c_index"])
while it.iternext():
    pass
assert it.finished
assert it.itersize == 4
assert it.ndim == 2
assert it.nop == 1
caught = 0
try:
    k = it.index
except ValueError:
    caught += 1
try:
    m = it.multi_index
except ValueError:
    caught += 1
assert caught == 2

plain = np.nditer(a)
missing = False
try:
    k = plain.index
except ValueError:
    missing = True
assert missing
