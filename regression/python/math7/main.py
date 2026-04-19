import math

x = math.sqrt(2.0)
y = math.cos(x)
z = math.sin(y)

# known numeric range
assert y > 0.15 and y < 0.16
assert z > 0.15 and z < 0.16
