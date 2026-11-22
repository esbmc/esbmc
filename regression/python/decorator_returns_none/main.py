def dec(f):
    return None


@dec
def g():
    return "g"


try:
    g()
    assert False
except TypeError:
    pass
