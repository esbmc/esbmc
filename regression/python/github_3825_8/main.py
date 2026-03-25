# Issue #3825: cross-type list ordering, bool vs int
a = [True]   # bool, stored as size=1
b = [2]      # int

# True == 1 < 2
assert a < b
assert a <= b
assert b > a
assert b >= a

# bool True == int 1
c = [1]
assert not (a < c)
assert not (a > c)
assert a <= c
assert a >= c

# bool False < int 1
d = [False]
assert d < c
assert d <= c
assert c > d
assert c >= d
