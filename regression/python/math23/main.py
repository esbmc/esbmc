import math

a = 2.0
b = 3.0

# log(a*b) = log(a) + log(b)
left = math.log(a * b)
right = math.log(a) + math.log(b)
assert left > right - 0.001 and left < right + 0.001

# log(a/b) = log(a) - log(b)
left2 = math.log(a / b)
right2 = math.log(a) - math.log(b)
assert left2 > right2 - 0.001 and left2 < right2 + 0.001
