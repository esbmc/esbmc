# Issue #10494: len(x) - k was computed unsigned and wrapped.
l = [1, 2, 3]
assert len(l) - 4 >= 0
