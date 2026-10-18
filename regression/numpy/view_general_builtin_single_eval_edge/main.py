import numpy as np

calls = 0

def lower():
    global calls
    calls = calls + 1
    return 1

a = np.array([3, 1, 4, 1, 5])
s = sorted(a[lower():4])
assert s == [1, 1, 4]
l = list(a[lower():3])
assert l == [1, 4]
assert calls == 2
