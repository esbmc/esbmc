# Issue #3825: cross-type list ordering, float LHS vs int RHS
a = [1.0, 2.0]
b = [1, 3]

# 1.0 == 1, then 2.0 < 3
assert a < b
assert a <= b
assert b > a
assert b >= a

# Equal across types (reverse of github_3825_6)
c = [1, 2]
assert not (a < c)
assert not (a > c)
assert a <= c
assert a >= c
