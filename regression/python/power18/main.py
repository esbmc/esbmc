import math

# Zero raised to various powers
assert 0.0 ** 1 == 0.0
assert 0.0 ** 2 == 0.0
assert 0.0 ** 0 == 1.0

# Infinity raised to various powers
assert math.isinf((math.inf) ** 1)
assert math.isinf((math.inf) ** 2)
assert (math.inf) ** -1 == 0.0
assert (-math.inf) ** 3 == -math.inf
