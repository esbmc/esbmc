# `int` bound inside another function is local to it, so the keyword rewrite
# still applies here; an unrelated `global` declaration does not disable it
# either (#7557).
counter = 0


def helper() -> int:
    int = 5
    return int


def bump() -> None:
    global counter
    counter = counter + 1


def main() -> None:
    bump()
    assert helper() == 5
    assert int("10", base=2) == 2
    assert counter == 1


main()
