import math

# exact identity
assert math.cos(0.0) == 1.0
assert math.sin(0.0) == 0.0

# sin(pi/2) ≈ 1
x = math.sin(math.pi / 2)
assert x > 0.999 and x < 1.001

# cos(pi/2) ≈ 0
y = math.cos(math.pi / 2)
assert y > -0.001 and y < 0.001
