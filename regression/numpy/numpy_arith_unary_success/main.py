import math
import numpy as np

x = np.array([-2, 0, 3])
n = np.negative(x)
assert n[0] == 2 and n[2] == -3
p = np.positive(x)
assert p[0] == -2 and p[2] == 3
f = np.absolute(np.array([-2.5, -0.0]))
assert f[0] == 2.5 and f[1] == 0.0 and math.copysign(1.0, f[1]) == 1.0
o = np.array([9, 9, 9])
np.negative(x, where=[True, False, True], out=o)
assert o[0] == 2 and o[1] == 9 and o[2] == -3
m = np.absolute(np.array([-9223372036854775807 - 1]))
assert m[0] == -9223372036854775807 - 1
