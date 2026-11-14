# GitHub #10600
pairs = [(1, 2), (3, 4)]
swapped = nondet_bool()
if swapped:
    pairs = [(3, 4), (1, 2)]
(a, b), (c, d) = pairs
if swapped:
    assert a == 3 and b == 4 and c == 1 and d == 2
else:
    assert a == 1 and b == 2 and c == 3 and d == 4
