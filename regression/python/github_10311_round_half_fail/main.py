# #10311: round() of a float literal ties to even only on an exact half.
a = round(2.5000000000001)
assert a == 2
