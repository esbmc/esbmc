# Verdict half of github_4715_irep2_native_body_assert_sideeffect: the delegated
# statement must still assert the guard it was given, on a nondet input so the
# claim survives constant folding.
def double(n: int) -> int:
    return n * 2


def main() -> None:
    x: int = nondet_int()
    __ESBMC_assume(x == 3)
    assert double(x) == 7


main()
