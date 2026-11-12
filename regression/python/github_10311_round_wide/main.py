# #10311: round() of a float literal does not wrap at 32 bits.
a: int = round(3e9)
assert a == 3000000000
b: int = round(-3e9)
assert b == -3000000000
