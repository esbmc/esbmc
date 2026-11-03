def main() -> None:
    n = nondet_int()
    if 0xD800 <= n <= 0xDFFF:
        c = chr(n)
        assert ord(c) != 0xDFFF


main()
