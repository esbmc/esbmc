# get(flag)() calls 42 when flag is false, which raises TypeError.
def g():
    return "s"


def k():
    return 42


def get(flag):
    if flag:
        a = g
    else:
        a = k()
    return a


flag = nondet_bool()
try:
    assert get(flag)() == "s"
    assert flag
except TypeError:
    assert not flag
