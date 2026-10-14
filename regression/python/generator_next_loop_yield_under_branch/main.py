# The only yield is in a loop, so once the setup has run, a next() under a
# branch still gets the right value.
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
    assert z == 2 or z == 3


main(nondet_bool())
