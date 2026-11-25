# Issue #10467: a non-literal list slice step was sliced with step 1.
l = [10, 20, 30, 40, 50]
s = 2
r = l[::s]
assert len(r) == 3
assert r[1] == 30
