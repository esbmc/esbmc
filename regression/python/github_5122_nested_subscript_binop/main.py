# Arithmetic over two constant double-subscripts of a nested list used to
# abort the frontend during the type-annotation pass: get_type_from_binary_expr
# read lhs["value"]["id"] assuming a single subscript, but for M[0][0] the
# subscripted value is itself a Subscript (M[0]) with no "id" key, tripping the
# nlohmann-json operator[] assertion (json.hpp) -> SIGABRT (#5122).

M = [[3.0, 4.0]]
d = M[0][0] + M[0][1]
assert d == 7.0

# Integer elements and other binary operators share the same path.
N = [[3, 4]]
e = N[0][0] * N[0][1]
assert e == 12

# Matrix shape from the issue's impact statement.
X = [[1.0, 2.0]]
W = [[3.0], [4.0]]
A = X[0][0] * W[0][0] + X[0][1] * W[1][0]
assert A == 11.0
