import numpy as np

a = np.array([3e38, 3e38], dtype=np.float32)
assert a.mean() == 3e38
