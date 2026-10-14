import numpy as np

v = np.angle(0 + 1j)
assert v > 1.5
assert v < 1.6
