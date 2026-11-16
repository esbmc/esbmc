def wrap(x):
    r = []
    for e in x:
        r.append(e)
    return r


def mk():
    return [[1]] + [[2]]


assert mk() == wrap([[1], [2]])
assert mk() != wrap([[1], [3]])
