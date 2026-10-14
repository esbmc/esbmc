# list(range(N)) with a compile-time-constant N must materialise real elements,
# not the nondet-garbage symbolic list (which raised a false dereference).
N = 10
l = list(range(N))
assert len(l) == 10
assert l[0] == 0
assert l[3] == 3
assert l[9] == 9

# A constant used as range() args in a nested position (list()) and with
# start/step folds too.
M = 3
r = list(range(1, 10, M))
assert r[0] == 1
assert r[1] == 4
assert len(r) == 3
