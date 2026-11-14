def main() -> None:
    s = nondet_str()
    if len(s) > 3:
        return
    try:
        if s[len(s)] == 'a':
            pass
        assert False
    except IndexError:
        pass


main()
