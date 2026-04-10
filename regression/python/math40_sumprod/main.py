import math

a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
result = math.sumprod(a, b)
assert math.fabs(result - 32.0) < 1e-12
