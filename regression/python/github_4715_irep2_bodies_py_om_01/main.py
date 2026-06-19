# Pins the Python list operational-model alloca path under --irep2-bodies.
# The list OM allocates element storage via __ESBMC_alloca(sizeof(struct)),
# whose allocated type comes from the size constant's #c_sizeof_type. If the
# body round-trip drops that type (PR #5330), the alloca degrades to a byte
# array and reading an element's value as a wide int trips an "Incorrect
# alignment" dereference failure. These reads must stay sound under the flag.
l = [10, 20, 30, 40, 50]
l.append(60)
assert l[0] == 10
assert l[2] == 30
assert l[5] == 60
