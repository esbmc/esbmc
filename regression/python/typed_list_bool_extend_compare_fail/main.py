a = [True, False]
a.extend([False])
expected = [True, False, True]
assert a == expected
