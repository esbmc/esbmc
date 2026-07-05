class Counter:
    count: int = 0


def advance(c: Counter) -> None:
    c.count += 5


def main() -> None:
    c = Counter()
    advance(c)
    # Wrong: the augmented assignment through the parameter ran, so count is 5.
    assert c.count == 0


main()
