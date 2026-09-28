def main() -> None:
    # format() with a spec that has width/align/sign/zero-pad but no
    # presentation type renders the value as str()/repr() (CPython's shortest
    # repr) and then pads. Previously any typeless float spec was rejected.
    assert format(1.5, "10") == "       1.5"
    assert format(1.5, "<10") == "1.5       "
    assert format(1.5, ">10") == "       1.5"
    assert format(1.5, "^10") == "   1.5    "
    assert format(1.5, "08") == "000001.5"
    assert format(-1.5, "10") == "      -1.5"
    assert format(-1.5, "08") == "-00001.5"
    assert format(1.5, "+") == "+1.5"
    assert format(1.5, "=+10") == "+      1.5"

    # The shortest-repr renderer folds inexact and exponential values too.
    assert format(0.30000000000000004, "25") == "      0.30000000000000004"
    assert format(1e-05, "10") == "     1e-05"
    assert format(1e16, "10") == "     1e+16"

    # A whole-number float keeps the ".0" suffix.
    assert format(2.0, "6") == "   2.0"
    assert format(100.0, "<8") == "100.0   "


main()
