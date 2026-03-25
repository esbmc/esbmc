a = 2
b = -10

result: float = a ** b
assert result < 0.001

eps = 1e-12
x = 10 ** (-6)
y = 0.000001

assert abs(x - y) < eps
