import math

assert math.pow(2.0, 10.0) == 1024.0
assert abs(math.pow(2.0, 0.5) - 1.4142135623730951) < 1e-15
assert math.gamma(5.0) == 24.0
assert abs(math.lgamma(0.5) - 0.5723649429247001) < 1e-13
