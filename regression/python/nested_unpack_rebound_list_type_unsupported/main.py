# A type-changing binding must not reuse the integer-tuple record.
pairs = [(1, 2)]
if nondet_bool():
    pairs = [("a", 2)]
(a, b), = pairs
assert b == 2
