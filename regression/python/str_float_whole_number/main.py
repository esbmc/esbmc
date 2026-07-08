def main() -> None:
    # CPython's str()/repr() of a whole-number float keeps the ".0" suffix:
    # str(1.0) == "1.0", not "1". Pre-fix the trailing-zero strip removed the
    # dot entirely, folding str(1.0) to "1".
    assert str(1.0) == "1.0"
    assert str(2.0) == "2.0"
    assert str(3.0) == "3.0"
    assert str(10.0) == "10.0"
    assert str(100.0) == "100.0"
    assert str(0.0) == "0.0"
    # A whole number with internal trailing zeros keeps them (the "." in the
    # %f output stops the trailing-zero strip): str(120.0) is "120.0", not "12".
    assert str(120.0) == "120.0"
    assert str(1000.0) == "1000.0"
    # Fractional floats are unchanged by the fix.
    assert str(0.5) == "0.5"
    assert str(0.1) == "0.1"
    assert str(1.5) == "1.5"
    assert str(5.5) == "5.5"
    # The trailing-digit guard leaves non-numeric spellings alone: no ".0"
    # is appended to "inf"/"nan".
    assert str(float("inf")) == "inf"
    assert str(float("nan")) == "nan"


main()
