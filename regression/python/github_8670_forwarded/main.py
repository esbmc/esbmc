# A str and an int read out of a mixed list parameter fill a str and an int
# parameter, whether spelt out or starred (#8670).
def wrap(text, cols):
    return text[:cols]


def spelt(d, e):
    assert wrap(d[0], d[1]) == e


def starred(d, e):
    assert wrap(*d) == e


spelt(["abcd", 2], "ab")
starred(["abcd", 3], "abc")
