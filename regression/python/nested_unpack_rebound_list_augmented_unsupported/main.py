pairs = [(1, 2)]
if nondet_bool():
    pairs += [(2, 1)]
(a, b), (c, d) = pairs
