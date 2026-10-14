# Companion to nondet_list23_fail: the elements are independent but still ints,
# so each is equal to itself and the length honours the requested bound.
x: list[int] = nondet_list(3)
assert len(x) >= 0
assert len(x) <= 3
if len(x) == 2:
    assert x[0] == x[0]
