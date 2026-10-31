import numpy as np

a = np.array(["ab", "c", ""])
assert a[0] == "ab" and a[1] == "c" and a[2] == ""
assert len(a[0]) == 2 and len(a) == 3
a[1] = "xyz"
assert a[1] == "xy"

s = np.array(["ab", "c", "def"])
tail = s[1:]
assert len(tail) == 2 and tail[0] == "c" and tail[1] == "def"
assert tail.itemsize == 12 and tail.nbytes == 24
every = s[::2]
assert every[1] == "def"
