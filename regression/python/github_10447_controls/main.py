def f(a):
    a[1:] = [3]


a = [1, 2]
a[1:] = [1, 2]
assert len(a) == 3
c = [1]
c[1:] = [3]
i = len(c) - 1
assert c[i] == 3
assert c[-1] == 3
d = [1]
f(d)
assert d[1] == 3
e = []
e[0:] = [3]
assert e[0] == 3
