# chr(n) above 127 folds to its multi-byte UTF-8 form; len() counts its code
# points, not its bytes (#7552).
def main() -> None:
    assert len(chr(200)) == 1
    assert len(chr(0x1F600)) == 1
    assert len(chr(200) + "a") == 2
    n: int = nondet_int()
    __ESBMC_assume(n >= 0 and n < 2)
    s = "ab"
    assert len(s[n] + "c") == 2


main()
