# r is 1 on the path where c is false.
def s():
    return "a"


def i():
    return 1


def call(h):
    return h()


k = call
c = nondet_bool()
r = k(s) if c else k(i)
assert r != 1
