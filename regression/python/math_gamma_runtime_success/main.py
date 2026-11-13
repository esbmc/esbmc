import math

x = nondet_float()
if x >= 3.0 and x <= 4.0:
    g = math.gamma(x)
    assert g >= 2.0 - 1e-9 and g <= 6.0 + 1e-9
    l = math.lgamma(x)
    assert l >= 0.69 and l <= 1.8
