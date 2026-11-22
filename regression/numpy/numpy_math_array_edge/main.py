import math
import numpy as np

e = np.log1p(np.array([]))
assert len(e) == 0
z = np.cbrt(np.array([-0.0, 0.0]))
assert math.copysign(1.0, z[0]) == -1.0
assert math.copysign(1.0, z[1]) == 1.0
s = np.expm1(np.array([-0.0]))
assert math.copysign(1.0, s[0]) == -1.0
b = np.arctanh(np.array([1.0, -1.0]))
assert b[0] == np.inf and b[1] == -np.inf
l = np.log1p(np.array([-1.0, -2.0]))
assert l[0] == -np.inf and np.isnan(l[1])
o = np.expm1(np.array([710.0, -710.0]))
assert o[0] == np.inf and o[1] >= -1.0 and o[1] < -0.99
