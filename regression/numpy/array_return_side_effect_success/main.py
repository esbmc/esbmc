import numpy as np

calls = 0

def side_effect():
    global calls
    calls = calls + 1
    return np.array([10, 20, 30])


a = side_effect()
assert calls == 1
assert a[0] == 10

b = side_effect()
assert calls == 2
assert b[1] == 20
