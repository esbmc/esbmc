a = 2
b = 4
c = 8

assert a * c + b == 20
assert a * (c + b) == 24

assert c // b - a == 0
assert c // (b - a) == 4

assert c % a + b == 4
assert c % (a + b) == 2

assert c ** a + b == 68
assert c ** (a + b) == 262144
