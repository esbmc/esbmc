import math

# sqrt(e^4) = e^2
x = math.sqrt(math.exp(4.0))
expected = math.exp(2.0)
assert x > expected - 0.01 and x < expected + 0.01

# exp(sqrt(4)) = e^2
y = math.exp(math.sqrt(4.0))
assert y > 7.38 and y < 7.40
