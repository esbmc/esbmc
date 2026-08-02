class Counter:
    count: int = 0
    step: int = 5


def advance(c: Counter) -> None:
    # Augmented assignment of a class-variable attribute through a `Class*`
    # parameter: reads the default, then writes back the instance member.
    c.count += c.step


def read_count(c: Counter) -> int:
    return c.count


def main() -> None:
    c = Counter()
    # Fresh instance sees the class-variable defaults through the parameter.
    assert read_count(c) == 0
    advance(c)
    assert c.count == 5
    advance(c)
    assert read_count(c) == 10
    # Non-zero default is initialised, not nondet.
    assert c.step == 5


main()
