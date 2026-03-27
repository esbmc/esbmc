import math

r = math.sqrt(4.0)
assert r == 2.0

s = math.sin(r)  # sin(2)
assert s > 0.90 and s < 0.92

c = math.cos(r)  # cos(2)
assert c > -0.42 and c < -0.41
