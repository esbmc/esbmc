# Issue #3825: list lexicographic comparison with integers
a = [1, 2, 3]
b = [1, 2, 4]

assert a < b
assert a <= b
assert b > a
assert b >= a

# Prefix comparison with ints
c = [1, 2]
assert c < a
assert a > c

# Equal lists
d = [1, 2, 3]
assert a <= d
assert a >= d
assert not (a < d)
assert not (a > d)

# Empty list
empty: list = []
assert empty < a
assert a > empty
