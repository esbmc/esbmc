# The recovered element type is really used: the read yields the list's value,
# so asserting a different one is detected.


def use(xs: list) -> int:
    return xs[0] + 0


def main() -> None:
    xs = [5, 3]
    assert use(xs) == 6


main()
