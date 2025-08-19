import math

x = float("inf")
assert math.isinf(x)

y = float("-inf")
assert math.isinf(y)

z = float(1.5)
assert z == 1.5
assert math.isinf(z) == False