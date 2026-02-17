import math as m

result = m.fsum([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
assert m.fabs(result - 1.0) < 1e-9
assert m.cbrt(27.0) == 3.0
