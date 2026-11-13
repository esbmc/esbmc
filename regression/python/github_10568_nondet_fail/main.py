def g():
    return 1


def func(flag):
    if flag:
        return g


a = func(nondet_bool())
try:
    a()
except TypeError:
    assert a is not None
