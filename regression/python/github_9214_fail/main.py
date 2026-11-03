# A scalar stored over a list element replaces the element (#9214).
x = [[1]]
x[0] = 5
assert x == [[1]]
