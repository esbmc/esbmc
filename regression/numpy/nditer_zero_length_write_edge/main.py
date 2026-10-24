import numpy as np

z = np.zeros(0, dtype=int)
count = 0
for x in np.nditer(z, op_flags=["readwrite"], flags=["zerosize_ok"]):
    x[...] = 1
    count += 1
assert count == 0
assert len(z) == 0

a = np.array([1, 2])
caught = False
try:
    np.nditer(a, op_flags=[["readwrite", "readonly"]])
except ValueError:
    caught = True
assert caught
