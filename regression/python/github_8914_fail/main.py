def f(n, items):
    w, v = items[0]
    return w + v


args = [100, [[60, 10], [50, 8]]]
assert f(args[0], args[1]) == 71
