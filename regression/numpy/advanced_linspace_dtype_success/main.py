import numpy as np

a = np.linspace(-1, 2, 4, dtype=np.int32)
assert a[0] == -1 and a[1] == 0 and a[3] == 2
