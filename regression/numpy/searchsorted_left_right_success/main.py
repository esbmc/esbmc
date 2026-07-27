import numpy as np

left = np.searchsorted(np.array([1, 3, 3, 5]), 3, side="left")
right = np.searchsorted(np.array([1, 3, 3, 5]), 3, side="right")

assert left == 1
assert right == 3
