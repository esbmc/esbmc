# 10462: sorted() of a list literal keeps the literal's element types.
s = sorted([2.5, 0.5])
assert type(s[0]) is float
assert s[0] == 0.5

t = sorted([2.5, 1.0])
assert t[0] == 1

u = sorted([-2.5, 0.5], reverse=True)
assert type(u[1]) is float
assert u[1] == -2.5

for x in reversed([2.5, 0.5]):
    assert type(x) is float
