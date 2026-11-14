l = [0.5, 1.5]
l.append(3)
x = l[2]
assert type(x) is int
assert x == 3

w = l[0]
assert type(w) is float

n = [1, 2]
n.append(True)
t = n[2]
assert type(t) is bool
