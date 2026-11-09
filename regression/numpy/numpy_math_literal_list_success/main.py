import numpy as np

e = np.expm1([0.0, 1.0])
assert e[0] == 0.0 and e[1] > 1.718 and e[1] < 1.719
l = np.log1p([[0.0], [1.0]])
assert l[0][0] == 0.0 and l[1][0] > 0.693 and l[1][0] < 0.694
c = np.cbrt((8.0, -27.0))
assert c[0] > 1.9999 and c[0] < 2.0001 and c[1] > -3.0001 and c[1] < -2.9999
t = np.arctanh([1.0, -0.0])
assert t[0] == np.inf and t[1] == 0.0
assert np.exp2([3.0])[0] > 7.999 and np.exp2([3.0])[0] < 8.001
assert np.arcsinh([0.0])[0] == 0.0 and np.arccosh([1.0])[0] == 0.0
