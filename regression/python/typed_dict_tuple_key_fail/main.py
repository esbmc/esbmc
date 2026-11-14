def main() -> None:
    a = nondet_int()
    b = nondet_int()
    d = {}
    d[a, b] = 1
    d[b, a] = 2
    assert d[a, b] == 1


main()
