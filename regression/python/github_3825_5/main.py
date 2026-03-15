# Issue #3825: list lexicographic comparison with mixed int and float (type_flag == 3)
a = [1, 2.5, 3]
b = [1, 3.0, 0]

# First differing element: 2.5 < 3.0
assert a < b
assert a <= b
assert b > a
assert b >= a

# Equal mixed lists
c = [1, 2.5, 3]
assert a <= c
assert a >= c
assert not (a < c)
assert not (a > c)
