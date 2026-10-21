# Equal int/str lists through parameters compare equal, unequal ones do not
# (#8898).
def same(x, y):
    assert x == y


def differ(x, y):
    assert x == y


same([10, "-"], [10, "-"])
differ([10, "-"], [10, "+"])
