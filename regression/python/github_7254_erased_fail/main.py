def compare(a, b):
    assert not (a < b)


def floats():
    compare(1.2, 1.9)


def ints():
    compare(9, 9)


floats()
ints()
