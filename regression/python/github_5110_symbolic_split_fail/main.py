# Issue #5110: split() on a symbolic receiver used to return a 1-element list,
# making this false assertion provable (unsound). Both possible receivers split
# into two parts, so the real length is 2 and the expected verdict is FAILED.
def main() -> None:
    i = nondet_int()
    __ESBMC_assume(0 <= i <= 1)
    s = ["a.b", "c.d"][i]
    assert len(s.split(".")) == 1


main()
