def g():
    return 1


def func(flag):
    if flag:
        return g
    return 42


def text(flag):
    if flag:
        return g
    return "s"


a = func(True)
assert a() == 1
b = func(False)
caught = False
try:
    b()
except TypeError:
    caught = True
assert caught

c = text(False)
try:
    c()
    assert False
except TypeError:
    pass
