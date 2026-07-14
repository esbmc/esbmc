import numpy as np

v = np.angle(-1 + 0j)
assert v > 3.1
assert v < 3.2
