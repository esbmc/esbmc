import math

v = 9.0
r = math.sqrt(v)

# sqrt(x)^2 ≈ x
check = r * r
assert check > 8.999 and check < 9.001

# combine with trig
t = math.sin(math.sqrt(1.0))  # sin(1)
assert t > 0.84 and t < 0.85
