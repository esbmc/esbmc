# The generator reads the global limit, which main's local limit hides, so
# its body cannot be inlined into main.
limit = 3


def gen():
    i = 0
    while i < limit:
        yield i
        i += 1


def main() -> None:
    limit = 1
    g = gen()
    x = next(g)
    y = next(g)
    assert y == 1
    assert limit == 1


main()
