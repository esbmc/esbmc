c = {1: 2, 2: "x"}
v = c[2]
assert isinstance(v, str)
assert v == "x"
w = c[1]
assert w * 3 == 6

for k in [1, 2]:
    u = c[k]
    assert isinstance(u, int) == (k == 1)

d = {1: 2}
d[1] = "x"
t = d[1]
assert t == "x"
