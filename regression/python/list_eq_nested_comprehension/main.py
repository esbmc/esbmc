# The == fast path decided this from the type map's record of the
# comprehension-built list instead of the list's runtime size, and reported
# the two lists unequal (#8100).
m = [[i * j for j in range(2)] for i in range(2)]
assert m == [[0, 0], [0, 1]]
