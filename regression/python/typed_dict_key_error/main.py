def main() -> None:
    a = nondet_int()
    b = nondet_int()
    d = {}
    d[a] = 5
    try:
        v = d[b]
        assert a == b
        assert v == 5
    except KeyError:
        assert a != b


main()
