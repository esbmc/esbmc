import numpy as np

a = np.array([-1.0, -2.0, -3.0])
np.fabs(a, where=[True, False, True], out=a)
assert a[0] == 1.0 and a[1] == -2.0 and a[2] == 3.0
o = np.zeros(3)
np.fabs(np.array([-4, 5]), out=o[:2])
assert o[0] == 4.0 and o[1] == 5.0 and o[2] == 0.0
