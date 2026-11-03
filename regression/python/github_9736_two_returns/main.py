def g(n):
    d = []
    d.append(n)
    if n > 5:
        return sorted(d)
    return d


assert g(7) == [7]
assert g(1) == [1]
assert isinstance(g(7)[0], int)
