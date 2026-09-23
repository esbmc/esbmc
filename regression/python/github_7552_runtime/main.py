# ord() decodes the UTF-8 chr() builds, for every code point (#7552).
def main() -> None:
    n: int = nondet_int()
    __ESBMC_assume(0 <= n and n <= 1114111)
    __ESBMC_assume(n < 55296 or n > 57343)
    assert ord(chr(n)) == n


main()
