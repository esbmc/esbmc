def dec(f):
    return 42


@dec
def g():
    return "g"


try:
    g()
    assert False
except TypeError:
    pass
