import numpy as np
import random

a = np.arange(6)
assert len(a[10:-10:-2]) == 3
assert len(a[0:4:-1]) == 0
n = random.randint(-8, 8)
if n == -7:
    assert len(a[n::-1]) == 0
if n == -1:
    assert len(a[:n:-1]) == 0
s = "abcdef"
assert s[4:0:-1] == "edcb"
