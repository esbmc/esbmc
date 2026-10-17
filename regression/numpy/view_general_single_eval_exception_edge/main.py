import numpy as np

calls = 0

def lower():
    global calls
    calls = calls + 1
    return 1

a = np.array([3, 1, 4, 1, 5])
s = sum(a[lower():4])
assert s == 6
assert calls == 1
try:
    m = min(a[lower():lower()])
except ValueError:
    pass
assert calls == 3
