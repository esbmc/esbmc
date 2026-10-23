def part(t):
    i, u = t
    return i + u[0]

def g(u):
    return [part((0, u))]

assert g([1, 1]) == [2]
