import string


def f(string):
    words = string.split()
    assert len(words) == 2


f("a b")
