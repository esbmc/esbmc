def main() -> None:
    k = nondet_int()
    d = {}
    d[k] = 1
    assert k in d
    assert len(d) == 1
    assert d[k] == 1


main()
