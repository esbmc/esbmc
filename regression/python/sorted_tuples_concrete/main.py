# sorted() over a list of constant integer tuples. The runtime sort model
# retypes tuple elements as int (so element access raised "int not
# subscriptable"). For a list whose elements are all constant integer tuples the
# sort is now done at convert time and a list of tuple literals is rebuilt, so
# the tuple element type is preserved and verification is cheap.

# Sorts lexicographically; tuple-level read-back.
s = sorted([(3, 1), (1, 2), (2, 9)])
assert s[0] == (1, 2)
assert s[1] == (2, 9)
assert s[2] == (3, 1)

# Equal first component: the second decides.
t = sorted([(1, 5), (1, 2), (1, 9)])
assert t[0] == (1, 2)
assert t[2] == (1, 9)

# Negative components.
u = sorted([(2, 0), (-1, 5), (0, 9)])
assert u[0][0] == -1
assert u[2][0] == 2

# reverse=True.
v = sorted([(1, 0), (3, 0), (2, 0)], reverse=True)
assert v[0] == (3, 0)
assert v[2] == (1, 0)

# Three-component tuples.
w = sorted([(1, 2, 3), (1, 1, 9), (1, 2, 0)])
assert w[0] == (1, 1, 9)
assert w[1] == (1, 2, 0)
