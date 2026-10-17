# pick is invoked through the alias get, and the function it returns gives
# str or int.
def s():
    return "a"


def i():
    return 1


def pick(flag):
    if flag:
        return s
    return i


get = pick
flag = nondet_bool()
h = get(flag)
r = h()
if not flag:
    assert isinstance(r, int)
