def main() -> None:
    # The builtin format(value) / format(value, "") with an empty spec on a
    # float previously errored ("format() spec '' is not supported for this
    # value"): py_format_number models only explicit float presentation types,
    # so the repr-like default landed on the reject. It now folds to str(value),
    # matching CPython. A negative literal is a UnaryOp(USub, Constant); the
    # extracted double carries the sign, so both branches fold correctly.

    # Whole-number floats keep CPython's ".0" suffix (format(1.0) == "1.0").
    assert format(1.0) == "1.0"
    assert format(-1.0) == "-1.0"
    assert format(100.0) == "100.0"
    assert format(-1000000.0) == "-1000000.0"
    # Signed zero keeps its sign: str(-0.0) == "-0.0".
    assert format(0.0) == "0.0"
    assert format(-0.0) == "-0.0"
    # Fractional floats fold to str(value) (only exactly-representable dyadic
    # values are asserted, for cross-platform stability).
    assert format(0.5) == "0.5"
    assert format(-2.5) == "-2.5"
    assert format(1.25) == "1.25"
    assert format(-0.125) == "-0.125"
    # An explicit empty spec is equivalent to no spec.
    assert format(1.5, "") == "1.5"
    # Ints and strings with an empty spec are unaffected.
    assert format(7) == "7"
    assert format(-3) == "-3"
    assert format("hi") == "hi"


main()
