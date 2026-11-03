x = 5
assert hash(x) == hash(x)
assert hash(5) == 5
assert hash(-1) == -2
assert hash(-5) == -5
assert hash(True) == 1
assert hash(2**61 - 1) == 0
assert hash(2**61) == 1
