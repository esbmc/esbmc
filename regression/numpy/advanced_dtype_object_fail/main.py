import numpy as np

a = np.array([1, "a"], dtype=np.object_)
assert a[1] == "a"
