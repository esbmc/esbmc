# Fail variant: the float broadcast result is asserted against a wrong value, so
# verification must fail. Exercises the IEEE-float ieee_add element-op path of
# build_scalar_broadcast (Part V Phase V.3).
import numpy as np

a = np.array([1.5, 2.5, 3.0])
b = a + 0.5
assert b[0] == 99.0  # actual value is 2.0
