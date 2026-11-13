# Issue #10494: len(x) - k was computed unsigned and wrapped.
l = [1, 2, 3]
assert len(l) - 4 < 0
assert -1 < len(l)
assert len(l) > -1
assert 4 - len(l) == 1
