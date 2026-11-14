def g():
    return 1


def func(flag):
    if flag:
        return g


a = func(False)
assert a is None
try:
    a()
    assert False
except TypeError:
    pass

b = func(True)
assert b is not None
assert b() == 1
r = a() if b is None else b()
assert r == 1
