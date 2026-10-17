def wrap(text, cols):
    return text[:cols]


def starred(d, e):
    assert wrap(*d) == e


starred(["abcd", 2], "abc")
