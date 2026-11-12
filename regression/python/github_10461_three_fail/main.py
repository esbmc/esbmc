# 10461: l[-2] on three mixed elements is the float.
l = [1]
l.append(2.5)
l.append(3)
assert type(l[-2]) is int
