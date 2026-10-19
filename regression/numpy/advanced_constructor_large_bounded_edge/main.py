import numpy as np

a = np.linspace(0, 999, 1000, dtype=np.int32)
assert len(a) == 1000
assert a[999] == 999 and a[500] == 500
