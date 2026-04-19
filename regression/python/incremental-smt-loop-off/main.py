def main() -> None:
    total: int = 0
    i: int = 0

    while i < 5:
        x: int = nondet_int()

        __VERIFIER_assume(-10 <= x <= 10)

        total += x
        assert total >= -50

        i += 1


main()
