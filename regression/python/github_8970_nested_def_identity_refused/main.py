# Each call of make_s makes a new inner, so a and b are two functions; one
# module-level function would make them the same.
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
a = pick(c)()
b = pick(c)()
assert a is not b
r = a()
