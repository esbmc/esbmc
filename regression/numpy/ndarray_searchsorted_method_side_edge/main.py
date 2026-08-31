import numpy as np

a = np.array([1, 3, 3, 3, 5])

left = a.searchsorted(3, side="left")
right = a.searchsorted(3, side="right")

assert left == 1
assert right == 4
