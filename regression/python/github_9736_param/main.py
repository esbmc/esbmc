def g(n):
    d = [n]
    return sorted(d)


a = g(2)
assert isinstance(a[0], int)
assert a == [2]
