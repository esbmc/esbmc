# inner reads x, a variable of make_s, so it is a new function on every call and
# cannot be one module-level function.
def make_s():
    x = "a"

    def inner():
        return x

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
