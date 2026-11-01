def f1(x):
    u = x[0]
    v = x * 3
    return u * v


def f5(x, i):
    t = x * 3
    v = x[i]
    return t[0] + v


a = f1([1, 2, 3])
assert len(a) == 9
assert f5(a, 2) == 5
