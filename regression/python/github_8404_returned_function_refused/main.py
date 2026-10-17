# pick returns a function returning str or one returning int, so the call
# through its result has no single type and is refused.
def s():
    return "a"


def i():
    return 1


def pick(flag):
    if flag:
        return s
    return i


flag = nondet_bool()
b = pick(flag)()
if flag:
    assert b == "a"
else:
    assert isinstance(b, int)
