# 10461: a negative constant index reads the element's own type.
l = [0.5]
l.append(3)
assert type(l[-1]) is int
assert l[-1] == 3
assert type(l[-2]) is float

p = [1, 2.5, 3]
p.pop()
assert type(p[-1]) is float
assert p[-1] == 2.5
