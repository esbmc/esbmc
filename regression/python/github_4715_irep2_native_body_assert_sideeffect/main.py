# A call in an assert or assume guard is a side effect, so convert_assert /
# convert_assume hand it to remove_sideeffects, which hoists it into a temp.
# The native arms declined that shape, taking the whole function to the
# round-trip; they now delegate the statement.
def double(n: int) -> int:
    return n * 2


def main() -> None:
    x: int = nondet_int()
    __ESBMC_assume(double(x) == 6)
    assert double(x) == 6
    y: int = double(x) + 1
    assert y == 7


main()
