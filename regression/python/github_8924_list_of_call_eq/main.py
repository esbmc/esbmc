def part(t):
    i, u = t
    return i + u[0]


def g(u):
    return [part((0, u))]


def h(v):
    return float(v)


def k():
    return [h(1)]


def mk():
    return [1, 1]


assert g([1, 1]) == [1]
assert g(mk()) == [1]
assert k() == [1]
assert [1] == k()
