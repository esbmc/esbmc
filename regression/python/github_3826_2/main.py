# Chained assignment: simple name + tuple target together
# z = (x, y) = (10, 20)

z = (x, y) = (10, 20)

assert x == 10
assert y == 20
assert z == (10, 20)
