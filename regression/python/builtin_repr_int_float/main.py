def main() -> None:
    # repr() of an int or a float previously routed to the general-call handler
    # ("Undefined function 'repr'"). It now folds to str(value), since
    # repr(x) == str(x) for ints and floats in Python (only str/container/object
    # reprs differ). str() already folded these; this closes the repr() side.

    # Integer literals (positive, negative, zero, large).
    assert repr(5) == "5"
    assert repr(-5) == "-5"
    assert repr(0) == "0"
    assert repr(1000000) == "1000000"
    assert repr(-1000000) == "-1000000"

    # Float literals: whole-number floats keep CPython's ".0" suffix.
    assert repr(1.0) == "1.0"
    assert repr(-1.0) == "-1.0"
    assert repr(0.0) == "0.0"
    assert repr(100.0) == "100.0"
    assert repr(-1000000.0) == "-1000000.0"
    # Fractional floats (exactly-representable dyadic values for stability).
    assert repr(1.5) == "1.5"
    assert repr(-2.5) == "-2.5"
    assert repr(0.25) == "0.25"
    assert repr(-0.125) == "-0.125"

    # A non-constant int routes to the runtime model, matching str(x).
    x: int = 40
    x = x + 2
    assert repr(x) == "42"

    # repr(x) is str(x) for numbers.
    assert repr(7) == str(7)
    assert repr(3.5) == str(3.5)


main()
