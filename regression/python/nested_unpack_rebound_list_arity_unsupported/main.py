pairs = [(1, 2)]
if nondet_bool():
    pairs = [(2, 1, 0)]
(a, b), = pairs
