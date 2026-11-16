def g(n):
    d = []
    d.append(n)
    d.append(n + 1)
    return sorted(d)


a = g(2)
assert isinstance(a[0], int)
assert a == [2, 3]
