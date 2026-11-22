# get(flag)() raises TypeError when flag is false.
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


r = get(nondet_bool())()
assert r == "s"
