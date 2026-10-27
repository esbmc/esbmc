# The value read through a str conditional key follows the chosen key (#9210).
d = {"a": 0, "b": 1}


def pick(k):
    return d[k]


c = nondet_bool()
r = pick("a" if c else "b")
assert r == (1 if c else 0)
