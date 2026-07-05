# Issue #5110: str.split(sep) must be sound on a symbolic / runtime string
# receiver, not just compile-time constants. Here the receiver is selected by a
# nondet index, so it cannot be constant-folded; split() must still return the
# correct number of parts and the correct elements (verified against CPython).
def main() -> None:
    i = nondet_int()
    __ESBMC_assume(0 <= i <= 1)
    s = ["a.b", "c.d"][i]
    parts = s.split(".")
    assert len(parts) == 2
    assert parts[0] == "a" or parts[0] == "c"
    assert parts[1] == "b" or parts[1] == "d"


main()
