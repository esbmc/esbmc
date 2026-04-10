import math

x = 3.5
# exp(log(x)) = x
result1 = math.exp(math.log(x))
assert result1 > 3.499 and result1 < 3.501

y = 2.0
# log(exp(y)) = y
result2 = math.log(math.exp(y))
assert result2 > 1.999 and result2 < 2.001
