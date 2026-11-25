# 10463: after the sort, l[0] is the int 1.
l = [2.5]
l.append(1)
l.sort()
assert type(l[0]) is float
