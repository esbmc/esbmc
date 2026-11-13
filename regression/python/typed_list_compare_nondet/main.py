def main() -> None:
    a = nondet_list(nondet_int)
    b = nondet_list(nondet_int)
    if len(a) > 2 or len(b) > 2:
        return
    if a == b:
        assert len(a) == len(b)
        if a:
            assert a[len(a) - 1] == b[len(b) - 1]
    else:
        assert len(a) != len(b) or a[0] != b[0] or a[1] != b[1]


main()
