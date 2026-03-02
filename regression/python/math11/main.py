import math

a = math.sqrt(9.0)

s = math.sin(a)
c = math.cos(a)

expr = s * s + c * c

# identity still holds after reuse
assert expr > 0.999 and expr < 1.001
