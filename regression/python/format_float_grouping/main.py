def main() -> None:
    # A typeless float spec with a ',' or '_' grouping option groups the
    # integer part of the str()/repr() rendering by 3; the fractional part is
    # left alone. Previously any float grouping spec was rejected.
    assert format(1234.5, ",") == "1,234.5"
    assert format(1000000.0, ",") == "1,000,000.0"
    assert format(-1234.5, ",") == "-1,234.5"
    assert format(1234567.89, ",") == "1,234,567.89"
    assert format(123456789.5, ",") == "123,456,789.5"
    assert format(1234.5, "_") == "1_234.5"
    assert format(1234567.0, "_") == "1_234_567.0"

    # Groups shorter than 3 digits are unchanged.
    assert format(999.0, ",") == "999.0"
    assert format(1000.0, ",") == "1,000.0"
    assert format(0.5, ",") == "0.5"

    # Grouping composes with sign, width and alignment.
    assert format(1234.5, "+,") == "+1,234.5"
    assert format(1234.5, "15,") == "        1,234.5"
    assert format(1234.5, "<15,") == "1,234.5        "

    # An exponential/inf/nan repr is left ungrouped, matching CPython.
    assert format(1.5e20, ",") == "1.5e+20"
    assert format(1e16, ",") == "1e+16"


main()
