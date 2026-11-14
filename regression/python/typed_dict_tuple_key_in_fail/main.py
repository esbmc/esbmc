def main() -> None:
    a = nondet_int()
    b = nondet_int()
    k = (a, b)
    d = {}
    d[a, 1] = 5
    assert k not in d


main()
