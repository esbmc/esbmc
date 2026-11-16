import math

assert math.gamma(-0.5) < 0.0
assert math.gamma(-1.5) > 0.0
assert math.isfinite(math.lgamma(-0.5))
assert math.isfinite(math.lgamma(-3.5))
assert math.lgamma(float("-inf")) == float("inf")
