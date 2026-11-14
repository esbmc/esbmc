# r is 1 on the path where c is false.
def s():
    return "a"


def i():
    return 1


def call(d, k):
    return d[k]()


key = "k"
c = nondet_bool()
d = {key: s if c else i}
r = call(d, key)
assert r == "a"
