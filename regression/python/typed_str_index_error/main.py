def main() -> None:
    s = nondet_str()
    if len(s) > 3:
        return
    n = len(s)
    try:
        if s[n] == 'a':
            pass
        assert False
    except IndexError:
        pass
    if n > 0:
        assert s[-1] == s[n - 1]


main()
