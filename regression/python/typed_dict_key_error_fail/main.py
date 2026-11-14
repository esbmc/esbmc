def main() -> None:
    a = nondet_int()
    d = {}
    d[1] = 5
    v = d[a]
    assert v == 5


main()
