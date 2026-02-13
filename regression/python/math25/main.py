import math

# e^5 ≈ 148.413
x = math.exp(5.0)
assert x > 148.0 and x < 149.0

# ln(100) ≈ 4.605
y = math.log(100.0)
assert y > 4.60 and y < 4.61

# exp(ln(50)) = 50
z = math.exp(math.log(50.0))
assert z > 49.99 and z < 50.01
