def main() -> None:
    s = nondet_str()
    if len(s) > 3:
        return
    if s[1] == 'a':
        pass


main()
