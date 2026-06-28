def main() -> None:
    # The builtin format(value, spec) was unmodelled. Fold the integer base
    # specs and the default/empty spec (str(value)) over a constant value.
    assert format(255, "x") == "ff"
    assert format(255, "X") == "FF"
    assert format(8, "o") == "10"
    assert format(5, "b") == "101"
    assert format(42, "d") == "42"

    # Default spec is str(value); negatives keep a leading '-'.
    assert format(42) == "42"
    assert format(-255, "x") == "-ff"
    assert format(0, "b") == "0"

    # A constant string formats to itself with the default spec.
    assert format("hi") == "hi"

    # The result is a str and composes (len / index).
    s = format(255, "x")
    assert len(s) == 2 and s[0] == "f"

    # The str.format() method is unaffected by the builtin handler.
    assert "{} {}".format(1, 2) == "1 2"


main()
