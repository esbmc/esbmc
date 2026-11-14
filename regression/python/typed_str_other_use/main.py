def main() -> None:
    s = nondet_str()
    if len(s) > 2:
        return
    t = s + "x"
    assert len(t) == len(s) + 1
    if len(s) > 0 and s[0] == 'a':
        assert t[0] == 'a'


main()
