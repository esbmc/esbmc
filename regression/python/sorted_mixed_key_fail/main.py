def main() -> None:
    # Mutually incomparable key values must not crash the preprocessor. The
    # arithmetic-key fold bails to the regular dispatch, which reports the same
    # clean mixed-type error a keyless sorted([1, "a"]) produces (CPython itself
    # raises TypeError here).
    x = sorted([1, "a"], key=lambda v: v)
    assert len(x) == 2


main()
