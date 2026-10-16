from math import tanh
import math as m

x = nondet_float()
if x == x:
    t = tanh(x)
    assert t >= -1.0 and t <= 1.0
    if x > 0.0:
        assert m.atan(x) <= 1.5707963267948966
