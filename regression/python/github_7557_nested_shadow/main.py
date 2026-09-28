# A nested scope sees the names its enclosing function binds, so the `pow`
# defined in outer() shadows the builtin inside inner() too. Binding these
# keywords to the builtin's parameter order would compute 16 (#7557).
def outer() -> int:
    def pow(exp: int, base: int) -> int:
        return exp - base

    def inner() -> int:
        return pow(base=2, exp=4)

    return inner()


def main() -> None:
    assert outer() == 2


main()
