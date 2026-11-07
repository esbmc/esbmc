def wrap(x):
    r = []
    for e in x:
        r.append(e)
    return r


def mk():
    a = []
    a.append([1, 2])
    a.append([3])
    return a


assert mk() == wrap([[1, 2], [3]])
assert mk() != wrap([[1, 2], [4]])
