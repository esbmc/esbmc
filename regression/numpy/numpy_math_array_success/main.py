import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 2.0)
a = np.array([x, 1.0])
r = np.log1p(a)
assert r[0] > 0.4 and r[0] < 1.1
assert r[1] > 0.69 and r[1] < 0.7
m = np.array([[x, -8.0], [0.0, 27.0]])
c = np.cbrt(m)
assert c[0][0] > 0.79 and c[0][0] < 1.26
assert c[0][1] > -2.0001 and c[0][1] < -1.9999
assert c[1][0] == 0.0
assert c[1][1] > 2.9999 and c[1][1] < 3.0001
