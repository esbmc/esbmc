def main() -> None:
    l = [1, 2, 3, 4]
    i = nondet_int()
    if 0 <= i <= 4:
        t = l[i:]
        u = t[::-1]
        assert len(u) != len(t)


main()
