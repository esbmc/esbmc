s = sorted([1, 1.0], reverse=True)
assert type(s[0]) is int
assert type(s[1]) is float

t = sorted([2.5, 1, 0.5, 3])
assert type(t[0]) is float and type(t[1]) is int
assert type(t[2]) is float and type(t[3]) is int
