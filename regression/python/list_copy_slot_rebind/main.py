m = [[1], [2]]
d = m[:]
d[0] = [5]
assert m[0] == [1]
assert d[0] == [5]

t = [(1, "a"), (2, "b")]
u = t.copy()
u[0] = (5, "z")
assert t[0][1] == "a"
assert u[0][1] == "z"

g = [{"a": 1}, {"a": 2}]
h = g[0:2]
h[0] = {"a": 5}
assert g[0]["a"] == 1
