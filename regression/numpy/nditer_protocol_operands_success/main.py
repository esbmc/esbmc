import numpy as np

a = np.array([5, 6, 7])
it = np.nditer(a, op_flags=[["readwrite"]])
assert it.operands[0][1] == 6
for x in it:
    x[...] = x * 2
assert it.operands[0][1] == 12
assert a[2] == 14
