# The sorted() shapes that already work: integer tuples, and a plain list of
# string-carrying tuples that is never sorted.

v = sorted([(2, 3), (1, 4)])
assert v[0][0] == 1
assert v[1][0] == 2

u = [(1, "a"), (2, "b")]
assert u[0][0] == 1
