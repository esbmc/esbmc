# Issue #9092: a raising if/while condition runs no branch and can be caught.
class C:
    pass


def main(x: int) -> None:
    c = C()
    hit = 0
    try:
        if len(c) > 0:
            hit = 1
        else:
            hit = 2
        hit = 3
    except TypeError:
        assert hit == 0
        hit = 4
    assert hit == 4
    try:
        while not len(c):
            hit = 5
        hit = 6
    except TypeError:
        hit = 7
    assert hit == 7
    try:
        while c.limit:
            hit = 8
        hit = 9
    except AttributeError:
        hit = 10
    assert hit == 10
    if x > 0:
        pass
    elif len(c) > 0:
        pass
    if False and len(c) > 0:
        hit = 11
    assert hit == 10


main(1)
