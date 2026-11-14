# Equal int and float elements keep their order, with and without reverse.
a = sorted([1, 1.0])
assert type(a[0]) is int and type(a[1]) is float

b = sorted([1.0, 1])
assert type(b[0]) is float and type(b[1]) is int

c = sorted([1, 2.5, 1.0], reverse=True)
assert c[0] == 2.5 and type(c[1]) is int and type(c[2]) is float

d = sorted([-1, -0.5, -2.0, 0])
assert d[0] == -2.0 and d[1] == -1 and d[2] == -0.5 and d[3] == 0
assert type(d[1]) is int and type(d[3]) is int

# Compared exactly: 2**53 + 1 is greater than the float 2.0**53.
e = sorted([9007199254740993, 9007199254740992.0])
assert type(e[0]) is float and e[1] == 9007199254740993
