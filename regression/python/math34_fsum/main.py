import math

values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
result = math.fsum(values)
assert math.fabs(result - 1.0) < 1e-12
