def main() -> None:
    i = nondet_int()
    d = {1: 10, 2: 20, 3: 30}
    if 1 <= i <= 3:
        assert d[i] == 10 * i
    d[2] = 0
    assert d[1] + d[2] + d[3] == 40


main()
