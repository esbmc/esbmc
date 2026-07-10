def main() -> None:
    # repr() of a float now folds to str(value), so this false assertion is a
    # real FAILED: repr(-1.0) is "-1.0", not "-1". Pre-fix repr() of a number
    # routed to the general-call handler ("Undefined function 'repr'") rather
    # than folding at all.
    assert repr(-1.0) == "-1"


main()
