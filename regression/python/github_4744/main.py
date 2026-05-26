from stubs import nondet_int, INT_BOUND


def main() -> None:
    a = nondet_int()
    assert a <= INT_BOUND or a > INT_BOUND


main()
