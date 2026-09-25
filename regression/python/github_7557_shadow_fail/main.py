# Python resolves pow when the call runs, so this def below main() shadows the
# builtin too. Binding the keywords to the builtin's own parameter order would
# compute 204 and prove it; the user function's order does not (#7557).
def main() -> None:
    assert pow(base=2, exp=4) == 204


def pow(exp: int, base: int) -> int:
    return exp * 100 + base


main()
