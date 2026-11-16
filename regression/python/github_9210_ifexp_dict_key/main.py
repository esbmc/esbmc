# A str conditional expression passed as a dict key types the parameter as str (#9210).
d = {"a": 0, "b": 1}


def pick(k):
    return d[k]


c = nondet_bool()
r = pick("a" if c else "b")
assert r == (0 if c else 1)
