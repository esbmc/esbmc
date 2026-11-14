def main() -> None:
    a = nondet_int()
    b = nondet_int()
    d = {}
    d[a, b] = 1
    d[b, a] = 2
    if a == b:
        assert d[a, b] == 2
    else:
        assert d[a, b] == 1
        assert d[b, a] == 2
    if b != a + 1:
        try:
            v = d[a, a + 1]
            assert False
        except KeyError:
            pass


main()
