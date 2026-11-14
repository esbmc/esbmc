def main() -> None:
    a = nondet_int()
    b = nondet_int()
    d = {}
    d[a] = 1
    d[b] = 2
    assert d[a] == 1


main()
