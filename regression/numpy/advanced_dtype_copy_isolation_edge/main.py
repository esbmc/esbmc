import numpy as np

a = np.array(["ab", "c"])
c = a.copy()
a[0] = "z"
assert c[0] == "ab" and a[0] == "z"
