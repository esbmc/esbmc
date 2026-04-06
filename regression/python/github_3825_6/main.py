# Issue #3825: cross-type list ordering (int list vs float list)
a = [1, 2]
b = [1.0, 3.0]

# 1 == 1.0, then 2 < 3.0
assert a < b
assert a <= b
assert b > a
assert b >= a

# Equal across types
c = [1.0, 2.0]
assert not (a < c)
assert not (a > c)
assert a <= c
assert a >= c
