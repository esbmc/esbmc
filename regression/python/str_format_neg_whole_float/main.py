def main() -> None:
    # A negative literal is a UnaryOp(USub, Constant); the empty "{}" field must
    # format it with str(), which keeps CPython's ".0" suffix on a whole-number
    # float: "{}".format(-1.0) is "-1.0", not "-1". Pre-fix the negative branch
    # used a raw ostream fold that dropped the ".0" (the positive branch already
    # kept it), so this was a false SUCCESSFUL.
    assert "{}".format(-1.0) == "-1.0"
    assert "{}".format(-2.0) == "-2.0"
    assert "{}".format(-100.0) == "-100.0"
    assert "{}".format(-1000000.0) == "-1000000.0"
    # Signed zero keeps its sign: str(-0.0) == "-0.0".
    assert "{}".format(-0.0) == "-0.0"
    # Negative fractional floats were already correct and stay so (only exactly
    # representable dyadic values are asserted for cross-platform stability).
    assert "{}".format(-0.5) == "-0.5"
    assert "{}".format(-1.5) == "-1.5"
    # Positive whole-number floats and non-float fields are unaffected.
    assert "{}".format(1.0) == "1.0"
    assert "x={} y={}".format(-3.0, -4) == "x=-3.0 y=-4"


main()
