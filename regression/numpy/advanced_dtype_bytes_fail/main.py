import numpy as np

a = np.array([b"ab"], dtype="S2")
assert a.itemsize == 2
