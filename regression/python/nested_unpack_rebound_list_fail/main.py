# GitHub #10600
pairs = [(1, 2)]
if nondet_bool():
    pairs = [(2, 1)]
(a, b), = pairs
assert a == 1
