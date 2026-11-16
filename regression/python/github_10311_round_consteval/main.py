# #10311: round() folded inside an assert ties to even only on an exact half.
assert round(2.5000000000001) == 3
assert round(-2.5000000000001) == -3
assert round(1.95, 1) == 1.9
assert round(-0.445, 2) == -0.45
assert round(9007199254740993, 0) == 9007199254740993
assert round(123456789012345666, -1) == 123456789012345670
assert round(-35, -1) == -40
assert round(True, 1) == 1
assert round(5000000000000000000, -19) == 0
assert round(-7, -25) == 0
y = round(123456789012345666, -1)
assert y == 123456789012345670
