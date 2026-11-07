def wrap(x):
    r = []
    for e in x:
        r.append(e)
    return r


def ints():
    return [1, 2]


assert ints() == wrap([1.0, 2.0])
assert ints() != wrap([1.0, 2.5])
