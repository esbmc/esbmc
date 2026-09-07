# A user function shadowing a builtin keeps its own parameter order: binding
# these keywords to the builtin's slots would swap them (#7557).
def pow(exp: int, base: int) -> int:
    return exp - base


def main() -> None:
    assert pow(base=2, exp=4) == 2


main()
