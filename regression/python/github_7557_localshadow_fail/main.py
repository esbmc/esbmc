# `int` bound locally in helper() does not shadow the builtin in main(), so
# base=2 reaches the call and int("10", base=2) is 2, not 10 (#7557).
def helper() -> int:
    int = 5
    return int


def main() -> None:
    assert int("10", base=2) == 10


main()
