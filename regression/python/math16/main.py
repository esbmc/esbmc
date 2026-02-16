import math

# ln(1) = 0
assert math.log(1.0) == 0.0

# ln(e) = 1
x = math.log(math.e)
assert x > 0.999 and x < 1.001

# ln(10) ≈ 2.303
y = math.log(10.0)
assert y > 2.30 and y < 2.31
