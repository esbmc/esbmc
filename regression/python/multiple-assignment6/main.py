a, *b = [1, 2, 3, 4]
assert a == 1
assert b == [2, 3, 4]

*c, d = [5, 6, 7]
assert c == [5, 6]
assert d == 7

