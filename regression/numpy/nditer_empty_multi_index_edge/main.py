import numpy as np

z = np.zeros((0, 2))
it = np.nditer(z, flags=["multi_index", "zerosize_ok"])
assert it.finished
count = 0
for x in it:
    count += 1
assert count == 0
assert it.itersize == 0

caught = False
try:
    np.nditer(z)
except ValueError:
    caught = True
assert caught
