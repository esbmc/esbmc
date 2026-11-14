# The dict value is s or i, so the call through it gives a str or an int.
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
assert isinstance(r, str) == c
if c:
    assert r == "a"
else:
    assert r + 1 == 2
