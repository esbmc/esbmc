# A list literal bound in one arm of an if is not what the other arm, or the
# code after the join, reads.
a = nondet_int()
l = [0]
if a > 0:
    l = [1]
else:
    assert l[0] == 0
    l = [2]
assert (l[0] == 2) == (a <= 0)
assert (l[0] == 1) == (a > 0)
