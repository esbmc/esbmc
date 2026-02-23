import math

# e^0 = 1
assert math.exp(0.0) == 1.0

# e^1 ≈ 2.718
x = math.exp(1.0)
assert x > 2.71 and x < 2.72

# e^2 ≈ 7.389
y = math.exp(2.0)
assert y > 7.38 and y < 7.40
