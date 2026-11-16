def wrap(x):
    r = []
    for e in x:
        r.append(e)
    return r


def mk():
    a = [[0], [1], [2]]
    return a[1:]


assert mk() == wrap([[1], [2]])
assert mk() != wrap([[1], [3]])
