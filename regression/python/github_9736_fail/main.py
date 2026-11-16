def g(n):
    d = []
    d.append(n)
    return sorted(d)


a = g(2)
assert isinstance(a[0], str)
