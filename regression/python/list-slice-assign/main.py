# List slice assignment: l[lower:upper:step] = src
# Covers: extended slice (step > 1), negative step (with and without
# explicit bounds), negative bounds, step-1 replacement with growth,
# shrinkage, insertion, deletion, self-assignment, and nested-list sharing.

# Extended slice, same length (CPython requires matching lengths)
l = [1, 2, 3, 4, 5, 6]
l[::2] = [10, 30, 50]
assert l == [10, 2, 30, 4, 50, 6]

# Negative step: indices 3, 2, 1, 0
p = [1, 2, 3, 4]
p[::-1] = [40, 30, 20, 10]
assert p == [10, 20, 30, 40]

# Step-1 grow: replace 2 elements with 4
m = [1, 2, 3, 4, 5]
m[1:3] = [9, 8, 7, 6]
assert m == [1, 9, 8, 7, 6, 4, 5]

# Step-1 shrink: replace 3 elements with 1
n = [1, 2, 3, 4, 5]
n[1:4] = [7]
assert n == [1, 7, 5]

# Negative bounds
q = [1, 2, 3, 4, 5]
q[-3:-1] = [8, 8]
assert q == [1, 2, 8, 8, 5]

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
