# 10461: l[-2] and l[-3] on three mixed elements.
l = [1]
l.append(2.5)
l.append(3)
assert type(l[-2]) is float
assert type(l[-3]) is int
assert l[-2] == 2.5 and l[-3] == 1
