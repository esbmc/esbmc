# Whether the first yield is consumed depends on c, so the second next() has
# no fixed position in the generator.
def gen():
    yield 1
    yield 2


def main(c: bool) -> None:
    g = gen()
    if c:
        next(g)
    x = next(g)
    assert x == 2


main(True)
