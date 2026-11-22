def main() -> None:
    i = nondet_int()
    d = {}
    d[i, i] = 1
    assert d[i, i] == 1


main()
