# 10463: after the reverse sort, l[0] is the int 1.
l = [1]
l.append(0.5)
l.sort(reverse=True)
assert type(l[0]) is float
