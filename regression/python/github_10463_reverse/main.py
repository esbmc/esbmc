# 10463: sort(reverse=True) reorders the int and float elements by value.
l = [1]
l.append(0.5)
l.sort(reverse=True)
assert l[0] == 1 and type(l[0]) is int
assert type(l[1]) is float
