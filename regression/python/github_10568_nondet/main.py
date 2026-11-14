def g():
    return 1


def func(flag):
    if flag:
        return g


a = func(nondet_bool())
try:
    assert a() == 1
except TypeError:
    assert a is None
