def main() -> None:
    a = nondet_list(nondet_int)
    b = nondet_list(nondet_int)
    if len(a) > 2 or len(b) > 2:
        return
    if a != b:
        assert len(a) != len(b)


main()
