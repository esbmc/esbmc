import numpy as np

a = np.array(["", ""])
assert a[0] == "" and len(a[1]) == 0
a[0] = "xyz"
assert a[0] == "x"
