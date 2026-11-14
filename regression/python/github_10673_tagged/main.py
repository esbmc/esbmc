# c(h=b) passes b to call, which returns what b returns.
def s():
    return "a"


def n():
    return 1


def call(h):
    return h()


c = call
b = s if nondet_bool() else n
r = c(h=b)
if b is s:
    assert r == "a"
else:
    assert r == 1
