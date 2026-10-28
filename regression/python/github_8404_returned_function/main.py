# pick returns a function returning str or one returning int, and its result
# is called directly, so each path sees its own result type.
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
