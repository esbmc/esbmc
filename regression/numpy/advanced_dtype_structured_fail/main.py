import numpy as np

a = np.zeros(2, dtype=[("x", np.int32), ("y", np.float64)])
assert a[0]["x"] == 0
