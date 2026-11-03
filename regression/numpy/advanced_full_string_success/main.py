import numpy as np

a = np.full(3, "ok")
assert a[2] == "ok"
a[0] = "no"
assert a[0] == "no" and a[1] == "ok"
