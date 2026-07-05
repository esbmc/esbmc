# List slice assignment: l[lower:upper:step] = src
# Covers: extended slice (step > 1), negative step, negative bounds,
# step-1 replacement with growth and shrinkage.
# Further cases (insertion, deletion, explicit negative-step bounds,
# self-assignment, nested-list sharing) live in list-slice-assign2,
# split to keep each test under the CI per-test timeout.

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

print("ok")
