import numpy as np

a = np.array(["ab", "cd", "ef"])
c = a
b = a[0:2]
c[0] = "zz"
assert b[0] == "zz"
