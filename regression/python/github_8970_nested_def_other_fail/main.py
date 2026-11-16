# As github_8970_nested_def, but the str path asserts what is false.
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
    assert r != "a"
