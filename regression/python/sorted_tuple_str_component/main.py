# Sorting by a string key, where string order differs from insertion order.
v = sorted([("pear", 1), ("apple", 2), ("fig", 3)])
assert v[0][0] == "apple"
assert v[1][0] == "fig"
assert v[2][0] == "pear"
assert v[0][1] == 2

# Tie on the first component: the second decides.
w = sorted([("a", 2), ("a", 1)])
assert w[0][1] == 1 and w[1][1] == 2

# Integer first component still orders numerically, not lexically.
u = sorted([(10, "x"), (9, "y")])
assert u[0][0] == 9 and u[1][0] == 10

# reverse= still applies.
r = sorted([(1, "a"), (2, "b")], reverse=True)
assert r[0][0] == 2
