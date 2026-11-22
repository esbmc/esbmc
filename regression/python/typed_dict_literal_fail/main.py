def main() -> None:
    i = nondet_int()
    d = {1: 10, 2: 20, 3: 30}
    if 1 <= i <= 3:
        assert d[i] != 20


main()
