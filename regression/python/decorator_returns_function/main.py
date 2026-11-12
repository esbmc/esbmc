def other():
    return "other"


def dec(f):
    return other


@dec
def g():
    return "g"


try:
    assert g() == "other"
except TypeError:
    assert False
