import string


def f(string):
    words = string.split()
    assert len(words) == 3


f("a b")
