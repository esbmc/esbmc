import numpy as np

a = np.full(2, 0, dtype=np.dtype([("a", "i4")]))
assert a[0]["a"] == 0
