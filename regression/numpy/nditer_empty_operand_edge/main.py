import numpy as np

z = np.zeros((0, 3))
row = np.zeros(3)
count = 0
for x, y in np.nditer([z, row], flags=["zerosize_ok"]):
    count += 1
assert count == 0

caught = False
try:
    np.nditer([z, row])
except ValueError:
    caught = True
assert caught
