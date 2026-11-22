import numpy as np

o = np.array([9.0, 9.0])
raised = False
try:
    np.remainder(np.array([1.0, 2.0, 3.0]), 2.0, out=o)
except ValueError:
    raised = True
assert raised and o[0] == 9.0 and o[1] == 9.0
calls = 0


def operand():
    global calls
    calls += 1
    return np.array([7, -7])


r = np.floor_divide(operand(), np.array([2, 2]))
assert calls == 1 and r[0] == 3 and r[1] == -4
