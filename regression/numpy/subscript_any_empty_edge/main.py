import numpy as np

a = np.array([10, 20, 30])
sel = a[[]]
assert len(sel) == 0
