# A binding of `int` local to another function still suppresses the keyword
# rewrite module-wide, so the keyword is dropped and base's default proves the
# wrong value. The same false proof as #7557, from a pattern that predates it.
def helper() -> int:
    int = 5
    return int


def main() -> None:
    assert int("10", base=2) == 10


main()
