import numpy as np

a = np.full(fill_value=2.5, dtype=np.int8, shape=(2,))
assert a[0] == 2 and a[1] == 2
