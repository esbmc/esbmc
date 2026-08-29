def check(n: int) -> None:
    assert n != 7


def main() -> None:
    x = nondet_int()
    __ESBMC_assume(0 <= x and x <= 10)
    check(x)


main()
