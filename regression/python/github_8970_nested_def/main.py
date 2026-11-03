# The functions pick returns return nested defs of str and int, and the result is
# called twice in one statement.
def make_s():
    def inner():
        return "a"

    return inner


def make_i():
    def other():
        return 1

    return other


def pick(c):
    if c:
        return make_s
    return make_i


c = nondet_bool()
r = pick(c)()()
if c:
    assert type(r) == str
    assert r.upper() == "A"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
