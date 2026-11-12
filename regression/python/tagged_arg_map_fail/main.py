def ident(x):
    return x


res = list(map(ident, [1, "a", 3.0]))
x, y, z = res
assert x == 1
assert y == "b"
assert z == 3.0
