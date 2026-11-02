log = []


def f(v):
    log.append(v)
    return len(log)


a, b = {1: 2.0}, f(10)
c, d = f(20), a
assert b == 1
assert c == 2
assert d == {1: 2.0}
assert log == [10, 20]
