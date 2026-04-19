import math

x = math.sin(math.cos(math.sqrt(1.0)))

# sqrt(1)=1, cos(1)≈0.5403, sin(0.5403)≈0.514
assert x > 0.51 and x < 0.52
