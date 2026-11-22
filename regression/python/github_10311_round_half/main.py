# #10311: round() of a float literal ties to even only on an exact half.
a = round(2.5000000000001)
assert a == 3
b = round(-2.4999999999999996)
assert b == -2
c = round(2.5)
assert c == 2
d = round(-3.5)
assert d == -4
