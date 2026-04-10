# Issue #3825: list lexicographic comparison with floats (type_flag == 1)
a = [1.0, 2.0, 3.0]
b = [1.0, 2.0, 4.0]

assert a < b
assert a <= b
assert b > a
assert b >= a

# Equal lists
c = [1.0, 2.0, 3.0]
assert a <= c
assert a >= c
assert not (a < c)
assert not (a > c)

# Prefix comparison
d = [1.0, 2.0]
assert d < a
assert a > d
