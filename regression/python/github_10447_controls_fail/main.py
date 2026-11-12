def f(a):
    a[1:] = [3]


a = [1, 2]
a[1:] = [1, 2]
c = [1]
c[1:] = [3]
d = [1]
f(d)
e = []
e[0:] = [3]
assert len(a) == 3 and c[-1] == 3 and d[1] == 3 and e[0] == 4
