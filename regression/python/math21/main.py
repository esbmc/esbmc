import math

# exp(0) = 1, sin(1) ≈ 0.841
x = math.sin(math.exp(0.0))
assert x > 0.84 and x < 0.85

# log(e^2) = 2, cos(2) ≈ -0.416
y = math.cos(math.log(math.exp(2.0)))
assert y > -0.42 and y < -0.41
