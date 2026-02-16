import math

# e^(-1) ≈ 0.368
x = math.exp(-1.0)
assert x > 0.36 and x < 0.37

# e^(-2) ≈ 0.135
y = math.exp(-2.0)
assert y > 0.13 and y < 0.14

# exp(-x) * exp(x) = 1
a = 1.5
product = math.exp(a) * math.exp(-a)
assert product > 0.999 and product < 1.001
