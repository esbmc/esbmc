import numpy as np

a = np.array(["ab", "cd"])
v = a[0:1]
v[0] = "zz"
assert a[0] == "ab"
