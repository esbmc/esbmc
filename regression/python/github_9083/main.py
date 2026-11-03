# chr() accepts a surrogate code point, as CPython does (#9083).
def main() -> None:
    n = nondet_int()
    if 0xD800 <= n <= 0xDFFF:
        c = chr(n)
        assert ord(c) == n


main()
