import math

r1 = math.nextafter(0.0, 1.0)
assert r1 > 0.0
r2 = math.nextafter(1.0, 2.0)
assert r2 > 1.0
