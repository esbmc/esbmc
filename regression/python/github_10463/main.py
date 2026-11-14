# 10463: sort() reorders the int and float elements by value.
l = [2.5]
l.append(1)
l.sort()
assert l[0] == 1 and type(l[0]) is int
assert type(l[1]) is float
