import numpy as np

calls = 0

def size():
    global calls
    calls = calls + 1
    return 3

f = np.full(size(), 1)
assert calls == 1
assert len(f) == 3
i = np.identity(2)
assert calls == 1
