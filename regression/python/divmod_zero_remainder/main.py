x = 100
y = 10
quotient, remainder = divmod(x, y)

assert quotient == 10
assert remainder == 0
assert x == quotient * y
