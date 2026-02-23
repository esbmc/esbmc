import math

x = 3.0
y = 4.0

h = math.sqrt(x*x + y*y)

# 3-4-5 triangle
assert h > 4.999 and h < 5.001

s = math.sin(h)
assert s < 0   # sin(5) ≈ -0.96
