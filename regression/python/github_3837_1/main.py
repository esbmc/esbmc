# Chained assignment with list comprehension: a = b = [i for i in range(3)]
a = b = [i for i in range(3)]
assert a == [0, 1, 2]
assert b == [0, 1, 2]
assert a[0] == 0
assert b[0] == 0
assert a[1] == 1
assert b[1] == 1
assert a[2] == 2
assert b[2] == 2
