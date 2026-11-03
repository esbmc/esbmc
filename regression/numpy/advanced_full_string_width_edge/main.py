import numpy as np

a = np.full(2, "abc")
a[0] = "abcdef"
a[1] = ""
assert a[0] == "abc" and len(a[0]) == 3
assert a[1] == ""
