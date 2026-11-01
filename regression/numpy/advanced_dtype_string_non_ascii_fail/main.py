import numpy as np

a = np.array(["\u00e9"])
assert len(a[0]) == 1
