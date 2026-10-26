def g(arr):
    for x in arr:
        yield g(x)

r = list(g([1]))
assert len(r) == 1
