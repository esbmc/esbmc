# github #8890: two wrapper levels over an int-or-str callee keep the str path.
def pick(k: bool):
    if k:
        return "s"
    return 3


def wrap(k: bool):
    return pick(k)


def wrap2(k: bool):
    return wrap(k)


k = nondet_bool()
d = wrap2(k)
assert d != "s"
