# Chained assignment with filtered list comprehension
a = b = [x * 2 for x in range(5) if x % 2 == 0]
assert a == [0, 4, 8]
assert b == [0, 4, 8]
assert len(a) == 3
assert len(b) == 3
