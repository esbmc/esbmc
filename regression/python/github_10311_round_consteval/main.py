# #10311: round() folded inside an assert ties to even only on an exact half.
assert round(2.5000000000001) == 3
assert round(-2.5000000000001) == -3
assert round(1.95, 1) == 1.9
assert round(-0.445, 2) == -0.45
assert round(9007199254740993, 0) == 9007199254740993
