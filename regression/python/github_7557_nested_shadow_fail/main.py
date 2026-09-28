# -2 is what these keywords compute if inner() fails to inherit the `pow`
# outer() binds: the rewrite then moves them into the builtin's parameter
# order, exp=2 and base=4, and the user function returns 2 - 4. CPython
# computes 4 - 2 (#7557).
def outer() -> int:
    def pow(exp: int, base: int) -> int:
        return exp - base

    def inner() -> int:
        return pow(base=2, exp=4)

    return inner()


def main() -> None:
    assert outer() == -2


main()
