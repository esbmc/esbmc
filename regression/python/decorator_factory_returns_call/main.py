import random


def factory(replace):
    def dec(f):
        if replace:
            return inner()
        return f

    def inner():
        return 42

    return dec


replace = random.random() < 0.5


@factory(replace)
def g():
    return "g"


try:
    a = g()
    assert not replace and a == "g"
except TypeError:
    assert replace
