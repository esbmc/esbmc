import numpy as np

a = np.arange(6)
assert len(a[4:0:-2]) != 2
