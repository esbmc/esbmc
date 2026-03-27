import math

a = 0.7

s = math.sin(a)
c = math.cos(a)

# sin² + cos² = 1
identity = s * s + c * c
assert identity > 0.999 and identity < 1.001
