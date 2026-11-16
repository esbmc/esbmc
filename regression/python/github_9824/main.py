def g():
    return 1


def func(flag):
    if flag:
        return g
    return 42


a = func(False)
try:
    a()
    assert False
except TypeError:
    pass

b = func(True)
assert b() == 1
