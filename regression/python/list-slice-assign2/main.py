# List slice assignment, part 2 (split from list-slice-assign to keep
# each test under the CI per-test timeout).
# Covers: insertion, deletion, negative step with explicit bounds,
# self-assignment, and nested-list sharing.

# Pure insertion (empty slice)
s = [1, 2, 3]
s[1:1] = [9]
assert s == [1, 9, 2, 3]

# Pure deletion (empty source)
r = [1, 2, 3]
r[1:2] = []
assert r == [1, 3]

# Negative step with explicit bounds: indices 4, 3, 2
u = [1, 2, 3, 4, 5]
u[4:1:-1] = [50, 40, 30]
assert u == [1, 2, 30, 40, 50]

# Self-assignment: src must be snapshotted before the list is mutated
t = [1, 2, 3]
t[1:] = t
assert t == [1, 1, 2, 3]

# Nested lists are stored by reference (shared, not deep-copied)
inner = [7, 8]
w = [[0], [1], [2]]
w[0:2] = [inner, [9]]
inner[0] = 99
assert w[0][0] == 99
assert w[1][0] == 9
assert w[2][0] == 2

print("ok")
