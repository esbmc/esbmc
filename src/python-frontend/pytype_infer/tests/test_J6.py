a = []

s1 = "x"
a.append(s1)

l1 = []
a.append(l1)

n1 = 1
a.append(n1)

s2 = "y"
a.append(s2)

assert isinstance(a[0], str)
assert isinstance(a[1], list)
assert isinstance(a[2], int)
assert isinstance(a[3], str)