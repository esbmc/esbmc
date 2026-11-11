import math
import numpy as np

a = np.array([0.5, 2.0, 3.0])
np.multiply(a, a, where=a > 1.0, out=a)
assert a[0] == 0.5 and a[1] == 4.0 and a[2] == 9.0
x = np.array([-4.0, 9.0])
o = np.zeros(2)
np.power(x, 0.5, where=x >= 0.0, out=o)
assert o[0] == 0.0 and abs(o[1] - 3.0) < 1e-12
z = np.zeros(1)
np.multiply(np.array([-0.0]), 1.0, out=z)
assert math.copysign(1.0, z[0]) == -1.0
