import numpy as np

a = np.array(["ab", "c", ""])
assert a[0] == "ab" and a[1] == "c" and a[2] == ""
assert len(a[0]) == 2 and len(a) == 3
a[1] = "xyz"
assert a[1] == "xy"
