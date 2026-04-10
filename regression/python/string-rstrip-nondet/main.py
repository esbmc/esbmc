def main() -> None:
    s = nondet_str()
    r = s.rstrip()

    assert len(r) <= len(s)

    if len(s) == 0:
        assert r == ""

    if len(s) > 0:
        last = s[len(s) - 1]
        if (
            last != " "
            and last != "\t"
            and last != "\n"
            and last != "\v"
            and last != "\f"
            and last != "\r"
        ):
            assert r == s

main()
