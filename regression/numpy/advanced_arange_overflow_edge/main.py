import numpy as np

a = np.arange(250, 260, dtype=np.uint8)
assert len(a) == 10
assert a[5] == 255 and a[6] == 0 and a[9] == 3
