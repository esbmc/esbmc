# 10463: lists of one numeric type keep their element types after sort().
a = [3]
a.append(1)
a.sort()
assert a[0] == 1 and type(a[0]) is int

b = [0.5]
b.append(2.5)
b.sort(reverse=True)
assert b[0] == 2.5 and type(b[0]) is float
