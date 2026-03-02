def main() -> None:
    s = nondet_str()
    u = s.upper()

    assert len(u) == len(s)

    if len(s) == 0:
        assert u == ""


main()
