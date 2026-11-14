def main() -> None:
    a = nondet_int()
    k = (a, 1)
    d = {}
    assert k not in d
    d[a, 1] = 5
    assert k in d
    assert (a, 2) not in d
    assert d[k] == 5


main()
