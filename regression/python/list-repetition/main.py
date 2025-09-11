def foo() -> None:
    x = 1
    arr = [1] * x
    assert arr[0] == 1
    x = [1, 2, 3]
    n = len(x)
    z = [2] * n
    assert z[0] == 2


a = [1] * 5
assert a[0] == 1
assert a[1] == 1
assert a[2] == 1
assert a[3] == 1
assert a[4] == 1

b = [1.5] * 4
assert b[0] == 1.5
assert b[1] == 1.5
assert b[2] == 1.5
assert b[3] == 1.5

c = [float("inf")] * 5
assert c[0] == float("inf")
assert c[1] == float("inf")
assert c[2] == float("inf")
assert c[3] == float("inf")

n = 3
d = [2] * n
assert d[0] == 2
assert d[1] == 2
assert d[2] == 2

e = [1, 2, 3]
f = len(e)
g = [3] * f
assert g[0] == 3
assert g[1] == 3
assert g[2] == 3

h = [1.5] * f
assert h[0] == 1.5
assert h[1] == 1.5
assert h[2] == 1.5

i = [float("inf")] * f
assert i[0] == float("inf")
assert i[1] == float("inf")
assert i[2] == float("inf")
