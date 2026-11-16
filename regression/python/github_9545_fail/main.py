# Calls passing lists of different element types each get a copy of f (#9545).
def f(xs):
    return xs


r = f(["a", "b"])
assert r[1] == "b"
s = f([4, 5])
assert s[1] == 4
