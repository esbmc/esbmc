# Without the branch, z is the second yield, so the assertion fails.
def gen():
    n = 1
    while True:
        yield n
        n += 1


def main(c: bool) -> None:
    g = gen()
    x = next(g)
    if c:
        y = next(g)
        assert y == 2
    z = next(g)
    assert x == 1
    assert z == 3


main(nondet_bool())
