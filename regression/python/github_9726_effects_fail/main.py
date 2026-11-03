log = []


def f(v):
    log.append(v)
    return len(log)


a, b = {1: 2.0}, f(10)
c, d = f(20), a
assert log == [20, 10]
