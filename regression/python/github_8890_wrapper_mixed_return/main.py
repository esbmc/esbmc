# github #8890: the same int-or-str callee, called from a top-level wrapper.
def pick(flag):
    if flag:
        return "s"
    return 3


def wrap(flag):
    return pick(flag)


c = nondet_bool()
d = wrap(c)
if c:
    assert d == "s"
else:
    assert d == 3
