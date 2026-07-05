# Float numeric-array +/* scalar broadcast: exercises the IEEE-float element op
# (ieee_add/ieee_mul) path of handle_array_operations' build_scalar_broadcast,
# distinct from the integer add/mul path (Part V Phase V.3).
import numpy as np

a = np.array([1.5, 2.5, 3.0])
b = a + 0.5
assert b[0] == 2.0
assert b[1] == 3.0
assert b[2] == 3.5

c = a * 2.0
assert c[0] == 3.0
assert c[1] == 5.0
assert c[2] == 6.0
