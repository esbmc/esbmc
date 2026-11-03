def g(s):
    d = []
    d.append(s)
    d.append("a")
    return sorted(d)


a = g("b")
assert isinstance(a[0], str)
assert a == ["a", "b"]
