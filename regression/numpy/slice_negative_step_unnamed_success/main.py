import numpy as np
import random

a = np.arange(6)
assert a[::-2][0] == 5
assert a[::-2][2] == 1
assert len(a[::-2]) == 3
assert a[4:0:-1][0] == 4
assert len(a[4:0:-1]) == 4
assert len(a[-2::-3]) == 2
n = random.randint(-8, 8)
if n == 4:
    assert len(a[n::-2]) == 3
    assert a[n:0:-3][1] == 1
