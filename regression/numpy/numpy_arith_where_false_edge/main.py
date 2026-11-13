import numpy as np

a = np.array([1.0, 2.0])
b = np.array([0.0, 4.0])
o = np.array([-1.0, -1.0])
np.divide(a, b, where=b != 0.0, out=o)
assert o[0] == -1.0 and o[1] == 0.5
np.multiply(a, b, where=False, out=o)
assert o[0] == -1.0 and o[1] == 0.5
m = np.array([[1, 2], [3, 4]])
n = np.zeros((2, 2), dtype=int)
np.add(m, np.array([10, 20]), where=np.array([True, False]), out=n)
assert n[0][0] == 11 and n[0][1] == 0 and n[1][0] == 13
