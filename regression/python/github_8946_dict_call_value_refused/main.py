from esbmc import nondet_bool


def f():
    return "a"


def g():
    return 1


def pick(c):
    if c:
        return f
    return g


c = nondet_bool()
d = {"k": pick(c)}
assert c or d["k"]() == 1
