# The generator's setup runs at the first next(), which only happens when c
# holds, so it cannot be placed inside that branch.
def gen():
    n = 1
    while True:
        yield n
        n += 1


def main(c: bool) -> None:
    g = gen()
    if c:
        next(g)
    y = next(g)
    assert y == 1 or y == 2


main(False)
