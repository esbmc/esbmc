def f():
    return "a"


def g():
    return 1


def call(d, k):
    return d[k]()


use_f = nondet_bool()
d = {"k": f if use_f else g}
r = call(d, "k")
if use_f:
    assert r == "a"
else:
    assert r == 2
