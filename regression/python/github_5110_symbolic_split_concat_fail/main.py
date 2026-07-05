# Issue #5110: a concatenation with a symbolic operand is not constant-foldable,
# so it exercises the runtime split model. "brand." + <uk|com> contains exactly
# one ".", so split(".") has length 2; asserting length 1 must FAIL.
def main() -> None:
    i = nondet_int()
    __ESBMC_assume(0 <= i <= 1)
    tlds = ["uk", "com"]
    s = "brand." + tlds[i]
    assert len(s.split(".")) == 1


main()
