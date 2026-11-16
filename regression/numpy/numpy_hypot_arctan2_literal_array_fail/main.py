import numpy as np

a = np.arctan2([1.0, -1.0], np.array([-1.0, -1.0]))
assert a[0] < 2.356
