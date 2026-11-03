def g(n):
    if n > 0:
        yield isinstance(g(n - 1), list)

r = list(g(1))
assert r == [True]
