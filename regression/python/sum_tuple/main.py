# sum() over a tuple must fold its members with Python's int->float
# promotion. The list-typed sum/sum_float models iterate a representation a
# tuple struct does not have, so they used to return garbage (even
# sum((1, 2, 3)) != 6). Cover all-int, all-float, mixed, start arg, empty,
# and a tuple bound to a variable.
assert sum((1, 2, 3)) == 6
assert sum((1.0, 2.5)) == 3.5
assert sum((1, 2.5, 3)) == 6.5
assert sum((1, 2, 3), 10) == 16
assert sum(()) == 0
assert sum((5,)) == 5
t = (1, 2.5, 3)
assert sum(t) == 6.5
