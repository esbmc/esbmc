import math

g = math.gamma(-0.5)
assert abs(g + 3.544907701811032) < 1e-12
l = math.lgamma(-0.5)
assert abs(l - 1.265512123484645) < 1e-12
assert math.lgamma(float("-inf")) == float("inf")
