def text():
    return "s"


def one():
    return 1


def call(a):
    return a()


c = nondet_bool()
x = call(text) if c else call(one)
assert isinstance(x, str) == c
