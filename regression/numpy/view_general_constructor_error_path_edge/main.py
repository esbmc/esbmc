import numpy as np

caught = False
try:
    f = np.full(-1, 0)
except ValueError:
    caught = True
assert caught
