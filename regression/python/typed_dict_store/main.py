def main() -> None:
    a = nondet_int()
    b = nondet_int()
    d = {}
    d[a] = 1
    d[b] = 2
    d[a] = d[a] + 10
    if a == b:
        assert d[a] == 12
    else:
        assert d[a] == 11
        assert d[b] == 2


main()
