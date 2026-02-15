import math

x = 0.8

s1 = math.sin(x)
s2 = math.sin(-x)

c1 = math.cos(x)
c2 = math.cos(-x)

# odd/even properties
assert s1 > 0 and s2 < 0
assert c1 > 0 and c2 > 0

assert s1 + s2 > -0.001 and s1 + s2 < 0.001
assert c1 - c2 > -0.001 and c1 - c2 < 0.001
