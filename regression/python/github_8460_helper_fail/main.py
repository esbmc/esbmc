import helper

string = "a b c"


def f(string):
    words = string.split()
    assert len(words) == 3


f("a b")
assert len(string.split()) == 3
assert helper.LETTERS == "ab"
