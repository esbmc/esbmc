# A failing assert ends the program even when more assertions follow it, so
# the out-of-bounds read after them is unreachable.
l = [0, 1]
i = nondet_int()
assert 0 <= i and i < 2
assert l[0] == 0
assert len(l) == 2
x = l[i]
