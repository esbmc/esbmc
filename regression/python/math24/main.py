import math

# exp(exp(0)) = exp(1) = e
x = math.exp(math.exp(0.0))
assert x > 2.71 and x < 2.72

# log(exp(exp(1))) = e
y = math.log(math.exp(math.exp(1.0)))
assert y > 2.71 and y < 2.72
