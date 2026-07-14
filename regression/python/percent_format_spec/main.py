def main() -> None:
    # printf-style % formatting now honours flags, width and precision (it
    # previously rejected anything beyond a bare conversion char, e.g. %.2f).
    assert "%.2f" % 3.14159 == "3.14"
    assert "%5d" % 42 == "   42"
    assert "%03d" % 7 == "007"
    assert "%-5d" % 42 == "42   "
    assert "%+d" % 5 == "+5"
    assert "%f" % 3.14 == "3.140000"
    assert "%8.2f" % 3.14159 == "    3.14"
    assert "%07.2f" % -3.1 == "-003.10"   # sign-aware zero padding
    assert "%e" % 1000.0 == "1.000000e+03"

    # Integer precision is a minimum digit count (zero-filled after sign).
    assert "%.3d" % 5 == "005"
    assert "%.3d" % -5 == "-005"
    assert "%.5x" % 255 == "000ff"
    assert "%8.3d" % 5 == "     005"

    # String width and precision (truncation).
    assert "%10s" % "hi" == "        hi"
    assert "%.3s" % "hello" == "hel"

    # Bare conversions are unchanged.
    assert "%s=%d" % ("x", 5) == "x=5"
    assert "%x" % 255 == "ff"


main()
