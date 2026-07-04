def main() -> None:
    # An empty "{}" replacement field formats a float with str(), which keeps
    # CPython's ".0" suffix on a whole-number float: "{}".format(1.0) is "1.0",
    # not "1". Pre-fix the ostream fold dropped the ".0".
    assert "{}".format(1.0) == "1.0"
    assert "{}".format(2.0) == "2.0"
    assert "{}".format(100.0) == "100.0"
    assert "{}".format(1000000.0) == "1000000.0"
    assert "{}".format(0.0) == "0.0"
    # Fractional floats are unchanged.
    assert "{}".format(0.5) == "0.5"
    assert "{}".format(3.14159) == "3.14159"
    assert "{}".format(1.5) == "1.5"
    # Multiple fields and non-float types are unaffected.
    assert "x={} y={}".format(3.0, 4) == "x=3.0 y=4"
    assert "{} {} {}".format(1, "a", True) == "1 a True"


main()
