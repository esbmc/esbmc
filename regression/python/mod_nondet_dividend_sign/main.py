def main() -> None:
    n = nondet_int() % 5
    assert n >= 0
    assert n < 5
    q = nondet_int()
    assert q // 5 * 5 + q % 5 == q


main()
