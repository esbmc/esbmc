import math
from math import sin, cos

y = math.cos(0)
assert y == 1.0

w = math.cos(0)
assert w == 1.0

angle = 1.5708
result = math.sin(angle)
assert result > 0.99 and result < 1.01

result2 = math.cos(angle)
assert result2 > -0.01 and result2 < 0.01

a = math.sin(math.cos(0))
assert a > 0.84 and a < 0.85

b = math.sin(1)
assert b > 0.84 and b < 0.85

c = math.sqrt(2)
d = math.sin(c)
assert d > 0.98 and d < 0.99
