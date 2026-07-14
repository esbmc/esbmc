def main() -> None:
    # str() of a constant float is folded to a char-array literal only when the
    # frontend can prove the 6-decimal spelling is CPython's exact repr, i.e. it
    # round-trips to the same double. Whole numbers keep the ".0" suffix and
    # finite dyadic/short-decimal values fold verbatim.
    assert str(2.5) == "2.5"
    assert str(0.25) == "0.25"
    assert str(0.5) == "0.5"
    assert str(0.1) == "0.1"
    assert str(12.75) == "12.75"
    assert str(100.0) == "100.0"
    assert str(1000.0) == "1000.0"
    assert str(0.0) == "0.0"
    assert str(-3.5) == "-3.5"
    # Non-finite floats keep CPython's spellings.
    assert str(float("inf")) == "inf"
    assert str(float("-inf")) == "-inf"
    assert str(float("nan")) == "nan"


main()
